"""Generate bounding box detections, then generate poses for patients
    """

import argparse
import glob
from glob import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import json
from predictor import VisualizationDemo
# import tcn_hpl.utils.utils as utils
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
import utils
import kwcoco
from mmpose.datasets import DatasetInfo
print(f"utils: {utils.__file__}")


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
    def __init__(self, config: dict):
        self.config = config
        self.root_path = config['root']
        # self.paths = utils.dictionary_contents(config['root'], types=['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.png'], recursive=True)
        
        # self.train_dataset = kwcoco.CocoDataset(config['data']['train'])
        # self.val_dataset = kwcoco.CocoDataset(config['data']['val'])
        # self.test_dataset = kwcoco.CocoDataset(config['data']['test'])
        
        self.dataset = kwcoco.CocoDataset(config['data'][config['task']])
        
        self.keypoints_cats = [
                        "nose", "mouth", "throat","chest","stomach","left_upper_arm",
                        "right_upper_arm","left_lower_arm","right_lower_arm","left_wrist",
                        "right_wrist","left_hand","right_hand","left_upper_leg",
                        "right_upper_leg","left_knee","right_knee","left_lower_leg", 
                        "right_lower_leg", "left_foot", "right_foot", "back"
                    ]
    
        self.keypoints_cats_dset = [{'name': value, 'id': index} for index, value in enumerate(self.keypoints_cats)]

        # self.train_dataset.dataset['keypoint_categories'] = self.keypoints_cats_dset
        # self.val_dataset.dataset['keypoint_categories'] = self.keypoints_cats_dset
        # self.test_dataset.dataset['keypoint_categories'] = self.keypoints_cats_dset
        
        self.dataset.dataset['keypoint_categories'] = self.keypoints_cats_dset
        
        self.dataset_path_name = self.config['data'][self.config['task']][:-12].split('/')[-1] #remove .mscoco.json
        
        self.args = utils.get_parser(self.config['detection_model_config']).parse_args()
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
        
    def generate_bbs_and_pose(self, dset, save_intermediate=True):
        
        # json_file = utils.initialize_coco_json()ds
        dsets_paths = []
        patient_cid = dset.add_category('patient')
        user_cid = dset.add_category('user')
        # pbar = tqdm.tqdm(enumerate(self.paths), total=len(self.paths))
        pbar = tqdm.tqdm(enumerate(dset.imgs.items()), total=len(list(dset.imgs.keys())))

        for index, (img_id, img_dict) in pbar:
            
            path = img_dict['file_name']
            
            img = read_image(path, format="BGR")
            predictions, visualized_output = self.predictor.run_on_image(img)
            
            instances = predictions["instances"].to('cpu')
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
            
            boxes = boxes.tensor.detach().numpy()
            scores = scores.numpy()
            
            file_name = path.split('/')[-1]
            # video_name = path.split('/')[-2]
            # print(f"file name: {file_name}")
            # print(f"file path: {path}")
            # print(f"video name: {video_name}")
            
            if boxes is not None:
                
                # person_results = []
                for box_id, _bbox in enumerate(boxes):

                    current_ann = {}
                    # current_ann['id'] = ann_id
                    current_ann['image_id'] = img_id
                    current_ann['bbox'] = np.asarray(_bbox).tolist()#_bbox
                    current_ann['category_id'] = patient_cid
                    current_ann['label'] = 'patient'
                    current_ann['bbox_score'] = str(round(scores[box_id] * 100,2)) + '%'
                    
                    
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
                    
                    # print(f"outputs: {type(returned_outputs[0])}")
                    # print(f"outputs: {len(returned_outputs)}")
                    # print(f"outputs: {returned_outputs[0]}")
                    image_features = returned_outputs[0]['backbone'][0,:,:,-1]
                    
                    # print(f"image_features: {image_features.shape}")
                    # exit()
                    pose_keypoints = pose_results[0]['keypoints'].tolist()
                    # bbox = pose_results[0]['bbox'].tolist()
                    pose_keypoints_list = []
                    for kp_index, keypoint in enumerate(pose_keypoints):
                        kp_dict = {'xy': [keypoint[0], keypoint[1]], 
                                'keypoint_category_id': kp_index, 
                                'keypoint_category': self.keypoints_cats[kp_index]}
                        pose_keypoints_list.append(kp_dict)
                    
                    current_ann['keypoints'] = pose_keypoints_list
                    # current_ann['image_features'] = image_features
                    
                    dset.add_annotation(**current_ann)
            
            if save_intermediate:
                if (index % 45000) == 0:
                    dset_inter_name = f"{self.config['data']['save_root']}/{self.dataset_path_name}_{index}_with_dets_and_pose.mscoco.json"
                    dset.dump(dset_inter_name, newlines=True)
                    print(f"Saved intermediate dataset at index {index} to: {dset_inter_name}")
                    
        return dset

    # def predict_poses(self, dset):
    #     pass
        
    def run(self):
        # self.train_dataset = self.generate_bbs_and_pose(self.train_dataset)
        # self.val_dataset = self.generate_bbs_and_pose(self.val_dataset)
        # self.test_dataset = self.generate_bbs_and_pose(self.test_dataset)
        
        self.dataset = self.generate_bbs_and_pose(self.dataset)
        
        # train_path_name = self.config['data']['train'][:-12].split('/')[-1] #remove .mscoco.json
        # val_path_name = self.config['data']['val'][:-12].split('/')[-1] #remove .mscoco.json
        # test_path_name = self.config['data']['test'][:-12].split('/')[-1] #remove .mscoco.json
        
        
        
        # train_path_with_pose = f"{self.config['data']['save_root']}/{train_path_name}_with_dets_and_pose.mscoco.json"
        # val_path_with_pose = f"{self.config['data']['save_root']}/{val_path_name}_with_dets_and_pose.mscoco.json"
        # test_path_with_pose = f"{self.config['data']['save_root']}/{test_path_name}_with_dets_and_pose.mscoco.json"
        
        dataset_path_with_pose = f"{self.config['data']['save_root']}/{self.dataset_path_name}_with_dets_and_pose.mscoco.json"
        # print(f"train_path: {train_path}")
        # print(f"train_path_with_pose: {train_path_with_pose}")
        # self.train_dataset.dump(train_path_with_pose, newlines=True)
        # print(f"Saved train dataset to: {train_path_with_pose}")
        
        # self.val_dataset.dump(val_path_with_pose, newlines=True)
        # print(f"Saved val dataset to: {val_path_with_pose}")
        
        # self.test_dataset.dump(test_path_with_pose, newlines=True)
        # print(f"Saved test dataset to: {test_path_with_pose}")
        
        self.dataset.dump(dataset_path_with_pose, newlines=True)
        print(f"Saved test dataset to: {dataset_path_with_pose}")

        

def main():
    
    main_config_path = f"configs/main.yaml"
    config = utils.load_yaml_as_dict(main_config_path)

    PG = PosesGenerator(config)
    PG.run()
    
if __name__ == '__main__':
    main()