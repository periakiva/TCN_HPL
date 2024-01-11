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
        self.paths = utils.dictionary_contents(config['root'], types=['*.JPG', '*.jpg', '*.JPEG', '*.jpeg', '*.png'], recursive=True)
        
        self.args = utils.get_parser().parse_args()
        detecron_cfg = setup_detectron_cfg(self.args)
        self.predictor = VisualizationDemo(detecron_cfg)
    
    def generate_bbs(self):
        
        json_file = utils.initialize_coco_json()
        
        pbar = tqdm.tqdm(enumerate(self.paths), total=len(self.paths))
        num_img, ann_num = 0, 0
        for index, path in pbar:
            img = read_image(path, format="BGR")
            predictions, visualized_output = self.predictor.run_on_image(img)
            
            instances = predictions["instances"].to('cpu')
            boxes = instances.pred_boxes if instances.has("pred_boxes") else None
            scores = instances.scores if instances.has("scores") else None
            classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None
            
            boxes = boxes.tensor.detach().numpy()
            scores = scores.numpy()
            
            file_name = path.split('/')[-1]
            video_name = path.split('/')[-2]
            print(f"file name: {file_name}")
            print(f"video name: {video_name}")
            
            if boxes is not None:
                num_img += 1
                # add images info
                current_img = {}
                # current_img['file_name'] = path.split('/')[-1]
                current_img['file_name'] = file_name
                current_img['id'] = num_img
                current_img['path'] = path
                current_img['video_name'] = video_name
                current_img['height'] = 720
                current_img['width'] = 1280
                json_file['images'].append(current_img)

                for box_id, _bbox in enumerate(boxes):


                    # add annotations
                    current_ann = {}
                    current_ann['id'] = ann_num
                    current_ann['image_id'] = current_img['id']
                    current_ann['bbox'] = np.asarray(_bbox).tolist()#_bbox
                    current_ann['category_id'] = classes[box_id] + 1
                    current_ann['label'] = 'patient'
                    current_ann['bbox_score'] = str(round(scores[box_id] * 100,2)) + '%'

                    if current_ann['category_id'] == 2:
                        continue
                    ann_num = ann_num + 1
                    json_file['annotations'].append(current_ann)
        return json_file

    def predict_poses(self, json_file):
        pass
        
    def run(self):
        json_file = self.generate_bbs()
        

def main():
    
    main_config_path = f"configs/main.yaml"
    config = utils.load_yaml_as_dict(main_config_path)

    PG = PosesGenerator(config)
    PG.run()
    
if __name__ == '__main__':
    main()