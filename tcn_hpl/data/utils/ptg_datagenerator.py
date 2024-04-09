# This script is designed for processing and preparing datasets for training activity classifiers.
# It performs several key functions: reading configuration files, generating feature matrices,
# preparing ground truth labels, and organizing the data into a structured format suitable for th TCN model.

import os
import yaml
import glob
import warnings
import kwcoco
import shutil

import numpy as np
import ubelt as ub

from pathlib import Path

from angel_system.data.medical.data_paths import GrabData
from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str,
)
from angel_system.activity_classification.train_activity_classifier import (
    data_loader,
    compute_feats,
)



def load_yaml_as_dict(yaml_path):
    """
    Loads a YAML file and returns it as a Python dictionary.

    Args:
        yaml_path (str): The path to the YAML configuration file.

    Returns:
        dict: The YAML file's content as a dictionary.
    """
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

#####################
# Inputs
#####################

# Mapping task identifiers to their descriptive names.
TASK_TO_NAME = {
    'm1': "M1_Trauma_Assessment",
    'm2': "M2_Tourniquet",
    'm3': "M3_Pressure_Dressing",
    'm4': "M4_Wound_Packing",
    'm5': "M5_X-Stat",
    'r18': "R18_Chest_Seal",
}

# Mapping lab data task identifiers to their descriptive names.
LAB_TASK_TO_NAME = {
    'm2': "M2_Lab_Skills",
    'm3': "M3_Lab_Skills",
    'm5': "M5_Lab_Skills",
    'r18': "R18_Lab_Skills",
}

# Mapping feature settings to boolean flags indicating the inclusion of pose or object joint information.
FEAT_TO_BOOL = {
        "no_pose": [False, False],
        "with_pose": [True, True],
        "only_hands_joints": [False, True],
        "only_objects_joints": [True, False]
    }


def main(task: str, ptg_root: str, config_root: str, 
         data_type: str, data_gen_yaml: str):
    """
    Main function that orchestrates the process of loading configurations, setting up directories,
    processing datasets, and generating features for training activity classification models.

    Args:
        task (str): The task identifier.
        ptg_root (str): Path to the root of the angel_system project.
        config_root (str): Path to the root of the configuration files.
        data_type (str): Specifies the type of data, either 'gyges' for professional data or 'bbn' for lab data.
    """
    config_path = f"{config_root}/experiment/{task}/feat_v6.yaml"
    config = load_yaml_as_dict(config_path)

    activity_config_path = f"{ptg_root}/config/activity_labels/medical"
    feat_version = 6

    #####################
    # Output
    #####################

    reshuffle_datasets, augment, num_augs = config['data_gen']['reshuffle_datasets'], config['data_gen']['augment'], config['data_gen']['num_augs']

    
    ## augmentation is not currently used.
    if augment:
        num_augs = num_augs
        aug_trans_range, aug_rot_range = [-5, 5], [-5, 5]
    else:
        num_augs = 1
        aug_trans_range, aug_rot_range = None, None

    # the "data_type" parameter is new to the BBN lab data. 
    # old experiments dont have that parameter in their experiment name
    """
    feat_type details wrt to feature generation:
        no_pose: we only use object detections, and object-hands intersection
        with_pose: we use patient pose to calculate joint-hands and joint-objects offset vectors
        only_hands_joints: we use patient pose to calculate only joint-hands offset vectors
        only_objects_joints: we use patient pose to calulcate only joint-objects offset vectors
    """
    
    exp_name = f"p_{config['task']}_feat_v6_{config['data_gen']['feat_type']}_v3_aug_{augment}_reshuffle_{reshuffle_datasets}_{data_type}" #[p_m2_tqt_data_test_feat_v6_with_pose, p_m2_tqt_data_test_feat_v6_only_hands_joints, p_m2_tqt_data_test_feat_v6_only_objects_joints, p_m2_tqt_data_test_feat_v6_no_pose]

    output_data_dir = f"{config['paths']['output_data_dir_root']}/{config['task']}/{exp_name}"

    gt_dir = f"{output_data_dir}/groundTruth"
    frames_dir = f"{output_data_dir}/frames"
    bundle_dir = f"{output_data_dir}/splits"
    features_dir = f"{output_data_dir}/features"

    # Create directories
    for folder in [output_data_dir, gt_dir, frames_dir, bundle_dir, features_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Clear out the bundles
    filelist = [f for f in os.listdir(bundle_dir)]
    for f in filelist:
        os.remove(os.path.join(bundle_dir, f))

    #####################
    # Mapping
    #####################
    activity_config_path = f"{config['paths']['activity_config_root']}/{config['task']}.yaml"
    with open(activity_config_path, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    # print(f"activity labels: {activity_labels}")
    activity_labels_desc_mapping = {}
    with open(f"{output_data_dir}/mapping.txt", "w") as mapping:
        for label in activity_labels:
            i = label["id"]
            label_str = label["label"]
            if "description" in label.keys():
                activity_labels_desc_mapping[label['description']] = label["label"]
            elif "full_str" in label.keys():
                activity_labels_desc_mapping[label['full_str']] = label["label"]
            if label_str == "done":
                continue
            mapping.write(f"{i} {label_str}\n")

    ###################
    # Features,
    # groundtruth and
    # bundles
    #####################
    ### create splits dsets
    
    # Data processing and feature generation based on the data type (gyges or bbn).
    # The detailed implementation handles data loading, augmentation (if specified),
    # ground truth preparation, feature computation, and data organization for training and evaluation.
    print(f"Generating features for task: {task}")
    
    if data_type == "pro":
        dset = kwcoco.CocoDataset(config['paths']['dataset_kwcoco'])
    elif data_type == "lab":
        dset = kwcoco.CocoDataset(config['paths']['dataset_kwcoco_lab'])

    if reshuffle_datasets:
        
        train_img_ids, val_img_ids, test_img_ids = [], [], []
        
        ## The data directory format is different for "Professional" and lab data, so we handle the split differently
        if data_type == "pro":
            task_name = config['task'].upper()
            train_vidids = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['train_vid_ids'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
            val_vivids = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['val_vid_ids'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
            test_vivds = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['test_vid_ids'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
            
            if config['data_gen']['filter_black_gloves']:
                vidids_black_gloves = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['names_black_gloves'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
                
                train_vidids = [x for x in train_vidids if x not in vidids_black_gloves]
                val_vivids = [x for x in val_vivids if x not in vidids_black_gloves]
                test_vivds = [x for x in test_vivds if x not in vidids_black_gloves]
                
            if config['data_gen']['filter_blue_gloves']:
                vidids_blue_gloves = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['names_blue_gloves'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
                
                train_vidids = [x for x in train_vidids if x not in vidids_blue_gloves]
                val_vivids = [x for x in val_vivids if x not in vidids_blue_gloves]
                test_vivds = [x for x in test_vivds if x not in vidids_blue_gloves]
        
        elif data_type == "lab":
            task_name = config['task'].upper()
            all_vids = sorted(list(dset.index.name_to_video.keys()))
            train_vidids = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['train_vid_ids_bbn']]
            val_vivids = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['val_vid_ids_bbn']]
            test_vivds = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['test_vid_ids_bbn']]

        for vid in train_vidids:
            if vid in dset.index.vidid_to_gids.keys():
                train_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
            else:
                print(f"{vid} not in the train dataset")

        for vid in val_vivids:
            
            if vid in dset.index.vidid_to_gids.keys():
                val_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
            else:
                print(f"{vid} not in the val dataset")

        for vid in test_vivds:
            
            if vid in dset.index.vidid_to_gids.keys():
                test_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
            else:
                print(f"{vid} not in the test dataset")

        train_img_ids, val_img_ids, test_img_ids = list(train_img_ids), list(val_img_ids), list(test_img_ids)

        print(f"train num_images: {len(train_img_ids)}, val num_images: {len(val_img_ids)}, test num_images: {len(test_img_ids)}")

        train_dset = dset.subset(gids=train_img_ids, copy=True)
        val_dset = dset.subset(gids=val_img_ids, copy=True)
        test_dset = dset.subset(gids=test_img_ids, copy=True)

    # again, both data types use different directory formatting. We handle both here
    data_grabber = GrabData(yaml_path=data_gen_yaml)
    if data_type == "pro":
        # skill_data_root = f"{config['paths']['bbn_data_dir']}/Release_v0.5/v0.56/{TASK_TO_NAME[task]}/Data"
        skill_data_root = f"{data_grabber.bbn_data_root}/{TASK_TO_NAME[task]}/Data"
    elif data_type == "lab":
        skill_data_root = f"{data_grabber.lab_bbn_data_root}/{LAB_TASK_TO_NAME[task]}"

    for dset, split in zip([train_dset, val_dset, test_dset], ["train_activity", "val", "test"]):
        for video_id in ub.ProgIter(dset.index.videos.keys()):
            
            video = dset.index.videos[video_id]
            video_name = video["name"]
            
            if "_extracted" in video_name:
                video_name = video_name.split("_extracted")[0]

            image_ids = dset.index.vidid_to_gids[video_id]
            num_images = len(image_ids)
            
            print(f"there are {num_images} in video {video_id}")

            video_dset = dset.subset(gids=image_ids, copy=True)
            
            # begin activity GT section
            # The GT given data is provided in different formats for the lab and professional data collections.
            # we handle both here.
            if data_type == "pro":
                video_root = f"{skill_data_root}/{video_name}"
                activity_gt_file = f"{video_root}/{video_name}.skill_labels_by_frame.txt"
            elif data_type == "lab":
                activity_gt_file = f"{skill_data_root}/{video_name}.skill_labels_by_frame.txt"
                if not os.path.exists(activity_gt_file):
                    print(f"activity_gt_file {activity_gt_file} doesnt exists. Trying a different way")
                    activity_gt_file = f"{skill_data_root}/{video_name}.txt"
            
            if not os.path.exists(activity_gt_file):
                print(f"activity_gt_file {activity_gt_file} doesnt exists. continueing")
                continue
            
            f = open(activity_gt_file, "r")
            text = f.read()
            f.close()
            
            activityi_gt_list = ["background" for x in range(num_images)]
            
            if data_type == "pro":
                text = text.replace('\n', '\t')
                text_list = text.split("\t")[:-1]
                for index in range(0, len(text_list), 3):
                    triplet = text_list[index:index+3]
                    # print(f"index: {index}, {text_list[index]}")
                    # print(f"triplet: {text_list[index:index+3]}")
                    start_frame = int(triplet[0])
                    end_frame = int(triplet[1])
                    desc = triplet[2]
                    gt_label = activity_labels_desc_mapping[desc]
                    
                    if end_frame-1 > num_images:
                        ### address issue with GT activity labels
                        print("Max frame in GT is larger than number of frames in the video")
                        
                    for label_index in range(start_frame, min(end_frame-1, num_images)):
                        activityi_gt_list[label_index] = gt_label
                    
            elif data_type == "lab":
                text = text.replace('\n', '\t')
                text_list = text.split("\t")#[:-1]
                for index in range(0, len(text_list), 4):
                    triplet = text_list[index:index+4]
                    start_frame = int(triplet[0])
                    end_frame = int(triplet[1])
                    desc = triplet[3]
                    gt_label = activity_labels_desc_mapping[desc]
                    
                    if end_frame-1 > num_images:
                        ### address issue with GT activity labels
                        print("Max frame in GT is larger than number of frames in the video")
                        
                    for label_index in range(start_frame, min(end_frame-1, num_images)):
                        # print(f"label_index: {label_index}")
                        activityi_gt_list[label_index] = gt_label

                # print(f"start: {start_frame}, end: {end_frame}, label: {gt_label}, activityi_gt_list: {len(activityi_gt_list)}, num images: {num_images}")
            
            import collections
            counter = collections.Counter(activityi_gt_list)
            # print(f"counter: {counter}")
            image_ids = [0 for x in range(num_images)]
            for index, img_id in enumerate(video_dset.index.imgs.keys()):
                im = video_dset.index.imgs[img_id]
                frame_index = int(im['frame_index'])
                video_dset.index.imgs[img_id]['activity_gt'] = activityi_gt_list[frame_index]
                dset.index.imgs[img_id]['activity_gt'] = activityi_gt_list[frame_index]
                image_ids[frame_index] = img_id
                
                # import matplotlib.pyplot as plt
                # image_show = dset.draw_image(gid=img_id)
                # plt.imshow(image_show)
                # plt.savefig(f'myfig_{task}_{img_id}.png')
                # if index >= 20:
                #     exit()
                # print(f"activity gt: {activityi_gt_list[frame_index]}")
                # print(dset.index.imgs[img_id])
                
                # exit()
            
            #### end activity GT section
            
            # features
            (
                act_map,
                inv_act_map,
                image_activity_gt,
                image_id_to_dataset,
                label_to_ind,
                act_id_to_str,
                ann_by_image,
            ) = data_loader(video_dset, activity_config)
            
            if split != "train_activity":
                num_augs = 1
                aug_trans_range, aug_rot_range = None, None
                
            for aug_index in range(num_augs):
                X, y = compute_feats(
                    act_map,
                    image_activity_gt,
                    image_id_to_dataset,
                    label_to_ind,
                    act_id_to_str,
                    ann_by_image,
                    feat_version=feat_version,
                    objects_joints=FEAT_TO_BOOL[config['data_gen']['feat_type']][0],
                    hands_joints=FEAT_TO_BOOL[config['data_gen']['feat_type']][1],
                    aug_trans_range = aug_trans_range,
                    aug_rot_range = aug_rot_range,
                    top_n_objects=3
                )

                X = X.T
                print(f"X after transpose: {X.shape}")

                if num_augs != 1:
                    video_name_new = f"{video_name}_{aug_index}"
                else:
                    video_name_new = video_name
                    
                npy_path = f"{features_dir}/{video_name_new}.npy"
                
                np.save(npy_path, X)
                print(f"Video info saved to: {npy_path}")

                # groundtruth
                with open(f"{gt_dir}/{video_name_new}.txt", "w") as gt_f, \
                    open(f"{frames_dir}/{video_name_new}.txt", "w") as frames_f:
                    for ind, image_id in enumerate(image_ids):
                        image = dset.imgs[image_id]
                        image_n = image["file_name"] # this is the shortened string

                        # frame_idx, time = time_from_name(image_n)
                        frame_idx = int(image['frame_index'])
                        # print(f"frame index: {frame_idx}, inds: {ind}")
                        # print(f"image_n: {image_n}")
                        
                        activity_gt = image["activity_gt"]
                        if activity_gt is None:
                            activity_gt = "background"

                        gt_f.write(f"{activity_gt}\n")
                        frames_f.write(f"{image_n}\n")

                # bundles
                with open(f"{bundle_dir}/{split}.split1.bundle", "a+") as bundle:
                    bundle.write(f"{video_name_new}.txt\n")

    print("Done!")
    print(f"Saved training data to {output_data_dir}")

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        help="Object detections in kwcoco format for the train set",
        type=str,
    )
    
    parser.add_argument(
        "--ptg-root",
        default='/home/local/KHQ/peri.akiva/angel_system',
        help="root to angel_system",
        type=str,
    )
    
    parser.add_argument(
        "--config-root",
        default="/home/local/KHQ/peri.akiva/projects/TCN_HPL/configs",
        help="root to TCN configs",
        type=str,
    )
    
    parser.add_argument(
        "--data-type",
        default="pro",
        help="pro=proferssional data, lab=use lab data",
        type=str,
    )
    
    parser.add_argument(
        "--data-gen-yaml",
        default="/home/local/KHQ/peri.akiva/projects/angel_system/config/data_generation/bbn_gyges.yaml",
        help="Path to data generation yaml file",
        type=str,
    )
    
    args = parser.parse_args()
    main(task=args.task, ptg_root=args.ptg_root, 
         config_root=args.config_root, data_type=args.data_type, 
         data_gen_yaml=args.data_gen_yaml)