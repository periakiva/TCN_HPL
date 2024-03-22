import argparse
import glob
from glob import glob
import os
import yaml

def get_parser(config_file):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=config_file,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file."
                        # , default='/shared/niudt/detectron2/images/Videos/k2/4.MP4'
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'"
        , default= ['/shared/niudt/DATASET/Medical/Maydemo/2023-4-25/selected_videos/new/M2-16/*.jpg'] # please change here to the path where you put the images
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window."
        , default='./bbox_detection_results'
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def load_yaml_as_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

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

def create_dir_if_doesnt_exist(dir_path):
    """
    This function creates a dictionary if it doesnt exist.
    :param dir_path: string, dictionary path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return


def initialize_coco_json():
    
    json_file = {}
    json_file['images'] = []
    json_file['annotations'] = []
    json_file['categories']  = []

    # get new categorie
    temp_bbox = {}
    temp_bbox['id'] = 1
    temp_bbox['name'] = 'patient'
    temp_bbox['instances_count'] = 0
    temp_bbox['def'] = ''
    temp_bbox['synonyms'] = ['patient']
    temp_bbox['image_count'] = 0
    temp_bbox['frequency'] = ''
    temp_bbox['synset'] = ''

    json_file['categories'].append(temp_bbox)

    temp_bbox = {}
    temp_bbox['id'] = 2
    temp_bbox['name'] = 'user'
    temp_bbox['instances_count'] = 0
    temp_bbox['def'] = ''
    temp_bbox['synonyms'] = ['user']
    temp_bbox['image_count'] = 0
    temp_bbox['frequency'] = ''
    temp_bbox['synset'] = ''

    json_file['categories'].append(temp_bbox)
    ann_num = 0
    
    return json_file