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

import warnings
warnings.filterwarnings("ignore")

def dictionary_contents(path: str, types: list, recursive: bool = False) -> list:
    """
    Extract files of specified types from directories, optionally recursively.

    Parameters:
        path (str): Root directory path.
        types (list): List of file types (extensions) to be extracted.
        recursive (bool, optional): Search for files in subsequent directories if True. Default is False.

    Returns:
        list: List of file paths with full paths.
    """
    files = []
    if recursive:
        path = path + "/**/*"
    for type in types:
        if recursive:
            for x in glob(path + type, recursive=True):
                files.append(os.path.join(path, x))
        else:
            for x in glob(path + type):
                files.append(os.path.join(path, x))
    return files

class PosesGenerator(object):
    def __init__(self, config: dict):
        self.config = config
        self.root_path = config['root']
        self.paths = dictionary_contents(config['root'], types=['*.JPG', '*.jpg', '*.JPEG', '*.jpeg'], recursive=True)
        
        self.predictor = VisualizationDemo(config['detectron_config'])
    
    def generate_bbs(self):
        pbar = tqdm.tqdm(enumerate(self.paths), total=len(self.paths))
        for index, path in pbar:
            img = read_image(path, format="BGR")
            predictions, visualized_output = demo.run_on_image(img)