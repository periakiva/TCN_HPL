"""Generate bounding box detections, then generate poses for patients
    """

import numpy as np
import warnings
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from tcn_hpl.data.utils.pose_generation.predictor import VisualizationDemo
# import tcn_hpl.utils.utils as utils
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from tcn_hpl.data.utils.pose_generation.utils import get_parser, load_yaml_as_dict
import kwcoco
from mmpose.datasets import DatasetInfo
# print(f"utils: {utils.__file__}")


import warnings
warnings.filterwarnings("ignore")


def setup_detectron_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


class PosesGenerator(object):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.root_path = config['root']
        
        if config['data_type'] == "bbn":
            self.config_data_key = "bbn_lab"
        else:
            self.config_data_key = "data"
            
        self.dataset = kwcoco.CocoDataset(config[self.config_data_key][config['task']])
        self.patient_cid = self.dataset.add_category('patient')
        self.user_cid = self.dataset.add_category('user')
        
        
        self.keypoints_cats = [
                        "nose", "mouth", "throat","chest","stomach","left_upper_arm",
                        "right_upper_arm","left_lower_arm","right_lower_arm","left_wrist",
                        "right_wrist","left_hand","right_hand","left_upper_leg",
                        "right_upper_leg","left_knee","right_knee","left_lower_leg", 
                        "right_lower_leg", "left_foot", "right_foot", "back"
                    ]
    
        self.keypoints_cats_dset = [{'name': value, 'id': index} for index, value in enumerate(self.keypoints_cats)]
        
        self.dataset.dataset['keypoint_categories'] = self.keypoints_cats_dset
        
        self.dataset_path_name = self.config[self.config_data_key][self.config['task']][:-12].split('/')[-1] #remove .mscoco.json
        
        self.args = get_parser(self.config['detection_model_config']).parse_args()
        detecron_cfg = setup_detectron_cfg(self.args)
        self.predictor = VisualizationDemo(detecron_cfg)
        
        self.pose_model = init_pose_model(config['pose_model_config'], 
                                     config['pose_model_checkpoint'], 
                                     device=config['device'])

        self.pose_dataset = self.pose_model.cfg.data['test']['type']
        self.pose_dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if self.pose_dataset_info is None:
            warnings.warn(
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
                DeprecationWarning)
        else:
            self.pose_dataset_info = DatasetInfo(self.pose_dataset_info)
    
    def predict_single(self, image: torch.tensor) -> list:
        
        predictions, _ = self.predictor.run_on_image(image)
        instances = predictions["instances"].to('cpu')
        boxes = instances.pred_boxes if instances.has("pred_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
        
        boxes_list, labels_list, keypoints_list = [], [], []
        
        if boxes is not None:
                
            # person_results = []
            for box_id, _bbox in enumerate(boxes):
                
                box_class = classes[box_id]
                if box_class == 0:
                    pred_class = self.patient_cid
                    pred_label = 'patient'
                elif box_class == 1:
                    pred_class = self.user_cid
                    pred_label = 'user'
                
                boxes_list.append(np.asarray(_bbox).tolist())
                labels_list.append(pred_label)
                
                
                current_ann = {}
                # current_ann['id'] = ann_id
                current_ann['image_id'] = 0
                current_ann['bbox'] = np.asarray(_bbox).tolist()#_bbox
                current_ann['category_id'] = pred_class
                current_ann['label'] = pred_label
                current_ann['bbox_score'] = f"{scores[box_id] * 100:0.2f}"
                
                if box_class == 0:
                    person_results = [current_ann]
                    
                    pose_results, returned_outputs = inference_top_down_pose_model(
                                                                                    model=self.pose_model,
                                                                                    img_or_path=image,
                                                                                    person_results=person_results,
                                                                                    bbox_thr=None,
                                                                                    format='xyxy',
                                                                                    dataset=self.pose_dataset,
                                                                                    dataset_info=self.pose_dataset_info,
                                                                                    return_heatmap=False,
                                                                                    outputs=['backbone'])
                    
                    pose_keypoints = pose_results[0]['keypoints'].tolist()
                    pose_keypoints_list = []
                    for kp_index, keypoint in enumerate(pose_keypoints):
                        kp_dict = {'xy': [keypoint[0], keypoint[1]], 
                                'keypoint_category_id': kp_index, 
                                'keypoint_category': self.keypoints_cats[kp_index]}
                        pose_keypoints_list.append(kp_dict)
                    
                    keypoints_list.append(pose_keypoints_list)
                    # print(f"pose_keypoints_list: {pose_keypoints_list}")
                    current_ann['keypoints'] = pose_keypoints_list
                    # current_ann['image_features'] = image_features
                
                # dset.add_annotation(**current_ann)
        
        # results = []
        return boxes_list, labels_list, keypoints_list
    
    def generate_bbs_and_pose(self, dset: kwcoco.CocoDataset, save_intermediate: bool =True) -> kwcoco.CocoDataset:
        
        """
        Generates a CocoDataset with bounding box (bbs) and pose annotations generated from the dataset's images.
        This method processes each image, detects bounding boxes and classifies them into 'patient' or 'user' categories,
        and performs pose estimation on 'patient' detections. Annotations are added to the dataset, including bounding
        box coordinates, category IDs, and, for patients, pose keypoints.

        Parameters:
        - dset (kwcoco.CocoDataset): The dataset to generate, which must be an instance of `kwcoco.CocoDataset`.
        - save_intermediate (bool, optional): If True, periodically saves the dataset to disk after processing a set number
        of images. This is useful for long-running jobs to prevent data loss and to track progress. Default is True.

        Returns:
        - kwcoco.CocoDataset: The input dataset, now added with additional annotations for bounding boxes and pose
        keypoints where applicable.

        Note:
        - The bounding box and pose estimation models are assumed to be accessible via `self.predictor` and `self.pose_model`,
        respectively. These models must be properly configured before calling this method.
        - The method uses a progress bar to indicate processing progress through the dataset's images.
        - This function automatically handles the categorization of detections into 'patient' and 'user' based on the model's
        predictions and performs pose estimation only on 'patient' detections.
        - Save intervals for the intermediate dataset dumps can be adjusted based on the dataset size and processing time
        per image to balance between progress tracking and performance.
        - The `kwcoco.CocoDataset` class is part of the `kwcoco` package, offering structured management of COCO-format
        datasets, including easy addition of annotations and categories, and saving/loading datasets.
        """
    
        # patient_cid = self.dataset.add_category('patient')
        # user_cid = self.dataset.add_category('user')
        pbar = tqdm.tqdm(enumerate(self.dataset.imgs.items()), total=len(list(self.dataset.imgs.keys())))
        
        for index, (img_id, img_dict) in pbar:
            
            path = img_dict['file_name']
            
            img = read_image(path, format="BGR")
            
            # bs, ls, kps = self.predict_single(img)
            
            # print(f"boxes: {bs}")
            # print(f"ls: {ls}")
            # print(f"kps: {kps}")
            
            # continue
            
            predictions, visualized_output = self.predictor.run_on_image(img)
            
            instances = predictions["instances"].to('cpu')
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
            
            boxes = boxes.tensor.detach().numpy()
            scores = scores.numpy()
            
            file_name = path.split('/')[-1]
            
            if boxes is not None:
                
                # person_results = []
                for box_id, _bbox in enumerate(boxes):
                    
                    box_class = classes[box_id]
                    if box_class == 0:
                        pred_class = self.patient_cid
                        pred_label = 'patient'
                    elif box_class == 1:
                        pred_class = self.user_cid
                        pred_label = 'user'
                        
                    current_ann = {}
                    # current_ann['id'] = ann_id
                    current_ann['image_id'] = img_id
                    current_ann['bbox'] = np.asarray(_bbox).tolist()#_bbox
                    current_ann['category_id'] = pred_class
                    current_ann['label'] = pred_label
                    current_ann['bbox_score'] = str(round(scores[box_id] * 100,2)) + '%'
                    
                    if box_class == 0:
                        person_results = [current_ann]
                        
                        pose_results, returned_outputs = inference_top_down_pose_model(
                                                                                        self.pose_model,
                                                                                        path,
                                                                                        person_results,
                                                                                        bbox_thr=None,
                                                                                        format='xyxy',
                                                                                        dataset=self.pose_dataset,
                                                                                        dataset_info=self.pose_dataset_info,
                                                                                        return_heatmap=None,
                                                                                        outputs=['backbone'])
                        
                        pose_keypoints = pose_results[0]['keypoints'].tolist()
                        pose_keypoints_list = []
                        for kp_index, keypoint in enumerate(pose_keypoints):
                            kp_dict = {'xy': [keypoint[0], keypoint[1]], 
                                    'keypoint_category_id': kp_index, 
                                    'keypoint_category': self.keypoints_cats[kp_index]}
                            pose_keypoints_list.append(kp_dict)
                        
                        # print(f"pose_keypoints_list: {pose_keypoints_list}")
                        current_ann['keypoints'] = pose_keypoints_list
                        # current_ann['image_features'] = image_features
                    
                    self.dataset.add_annotation(**current_ann)
            
            # import matplotlib.pyplot as plt
            # image_show = dset.draw_image(gid=img_id)
            # plt.imshow(image_show)
            # plt.savefig(f"figs/myfig_{self.config['task']}_{index}.png")
            # if index >= 20:
            #     exit()
            
            if save_intermediate:
                if (index % 45000) == 0:
                    dset_inter_name = f"{self.config[self.config_data_key]['save_root']}/{self.dataset_path_name}_{index}_with_dets_and_pose.mscoco.json"
                    self.dataset.dump(dset_inter_name, newlines=True)
                    print(f"Saved intermediate dataset at index {index} to: {dset_inter_name}")
                    
        return self.dataset
        
    def run(self) -> None:
        """
        Executes the process of generating bounding box and pose annotations for a dataset and then saves the
        enhanced dataset to disk.

        This method serves as the main entry point for the class it belongs to. It calls `generate_bbs_and_pose`
        with the current instance's dataset to add bounding box and pose annotations based on the results of object
        detection and pose estimation models. After processing the entire dataset, it saves the enhanced dataset
        with annotations to a specified location on disk in COCO format.

        The final dataset, including all generated annotations, is saved to a JSON file named according to the
        configuration settings provided in `self.config`, specifically within the 'save_root' directory and named
        to reflect that it includes detections and pose estimations.

        Note:
        - This method relies on `self.generate_bbs_and_pose` to perform the actual processing of the dataset, which
        must be properly implemented and capable of handling the dataset's images.
        - The save path for the final dataset is constructed from configuration parameters stored in `self.config`.
        - The method prints the path to the saved dataset file upon completion, providing a reference to the output.
        - It's assumed that `self.dataset` is already loaded or initialized and is an instance compatible with the
        processing expected by `generate_bbs_and_pose`.
        """
        self.dataset = self.generate_bbs_and_pose(self.dataset)
        
        dataset_path_with_pose = f"{self.config[self.config_data_key]['save_root']}/{self.dataset_path_name}_with_dets_and_pose.mscoco.json"        
        self.dataset.dump(dataset_path_with_pose, newlines=True)
        print(f"Saved test dataset to: {dataset_path_with_pose}")
        return

def main():
    
    main_config_path = f"configs/main.yaml"
    config = load_yaml_as_dict(main_config_path)

    PG = PosesGenerator(config)
    PG.run()
    
if __name__ == '__main__':
    main()