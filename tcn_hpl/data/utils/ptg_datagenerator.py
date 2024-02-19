import os
import yaml
import glob
import warnings
import kwcoco
import shutil

import numpy as np
import ubelt as ub

from pathlib import Path

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
from angel_system.data.data_paths import grab_data, data_dir

def load_yaml_as_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

#####################
# Inputs
#####################
task = "m2"
obj_exp_name = "p_bbn_model_m2_m3_m5_r18_v11" #"coffee+tea_yolov7"


#  f"{ptg_root}/config/activity_labels/medical"
# obj_dets_dir = f"{data_dir}/annotations/{recipe}/results/{obj_exp_name}"
# obj_dets_dir = "/data/PTG/cooking/object_anns/old_coffee/results/coffee_base/" #"/home/local/KHQ/hannah.defazio/yolov7/runs/detect/coffee+tea_yolov7/"
# obj_dets_dir = "/home/local/KHQ/peri.akiva/projects/medical-pose/bbox_detection_results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets.mscoco.json"
ptg_root = "/home/local/KHQ/peri.akiva/angel_system"
activity_config_path = f"{ptg_root}/config/activity_labels/medical"
# activity_config_fn = f"{activity_config_path}/task_{task}.yaml"
activity_config_fn = "/home/local/KHQ/peri.akiva/projects/angel_system/config/activity_labels/medical_tourniquet.v3.yaml"

feat_version = 6
using_done = False # Set the gt according to when an activity is done

#####################
# Output
#####################
reshuffle_datasets = True
augment = False
num_augs = 15
if augment:
    num_augs = num_augs
    aug_trans_range, aug_rot_range = [-15, 15], [-10, 10]
else:
    num_augs = 1
    aug_trans_range, aug_rot_range = None, None

feat_type = "with_pose" #[no_pose, with_pose, only_hands_joints, only_objects_joints]
feat_to_bools = {
    "no_pose": [False, False],
    "with_pose": [True, True],
    "only_hands_joints": [False, True],
    "only_objects_joints": [True, False]
}
exp_name = f"p_m2_tqt_data_test_feat_v6_{feat_type}_v2_aug_{augment}_reshuffle_{reshuffle_datasets}" #[p_m2_tqt_data_test_feat_v6_with_pose, p_m2_tqt_data_test_feat_v6_only_hands_joints, p_m2_tqt_data_test_feat_v6_only_objects_joints, p_m2_tqt_data_test_feat_v6_no_pose]
if using_done:
    exp_name = f"{exp_name}_done_gt"

output_data_dir = f"/data/PTG/TCN_data/m2/{exp_name}"

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
with open(activity_config_fn, "r") as stream:
    activity_config = yaml.safe_load(stream)
activity_labels = activity_config["labels"]

with open(f"{output_data_dir}/mapping.txt", "w") as mapping:
    for label in activity_labels:
        i = label["id"]
        label_str = label["label"]
        if label_str == "done":
            continue
        mapping.write(f"{i} {label_str}\n")

###################
# Features,
# groundtruth and
# bundles
#####################
### create splits dsets

vidids_black_gloves = [22,23,26,24,25,27,29,28,41,42,43,44,45,46,47,48,49,78,
                       79,84,88,90,80,81,82,83,85,86,87,89,91,99,110,111,121,113,115,116]
vidids_blue_gloves = [132,133,50,51,54,55,56,52,61,59,53,57,62,65,66,67,68,69,
                      58,60,63,64,125,126,127,129,131,134,135,136,128,130,137,
                      138,139]

filter_black_gloves, filter_blue_gloves = True, True

train_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_train_activity_obj_results_with_dets_and_pose.mscoco.json"
val_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_val_obj_results_with_dets_and_pose.mscoco.json"
test_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_test_obj_results_with_dets_and_pose.mscoco.json"

# train_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_train_activity_obj_results.mscoco.json"
# val_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_val_obj_results.mscoco.json"
# test_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_test_obj_results.mscoco.json"
train_dset = kwcoco.CocoDataset(train_file)
val_dset = kwcoco.CocoDataset(val_file)
test_dset = kwcoco.CocoDataset(test_file)

if reshuffle_datasets:
    dset = kwcoco.CocoDataset.union(train_dset,val_dset,test_dset)
    train_img_ids, val_img_ids, test_img_ids = [], [], []

    train_vidids = [1, 2, 4, 8, 9, 10, 11, 12, 16, 17,18, 20, 13, 19, 21, 30, 31, 32, 33, 34,35,36,38,
                    39,40,7,132,133,50,51,54,56,52,61,53,57,65,66,67,68,69,58,60,63,64,125,126,
                    127,129,131,134,135,136,119,122,124,70,71,72,92,93,94,95,97,98,100,
                    101,102,103,104,105,107,108,112,114,117,118,73,120,123,75,76,77]
    

    val_vivids= [5,6,37,59,106,130,138]
    
    test_vivds= [5,6,37,59,106,130,138]
    
    # test_vivds = [3,14,55,62,96,109,128,137,139]
    
    if filter_black_gloves:
        train_vidids = [x for x in train_vidids if x not in vidids_black_gloves]
        val_vivids = [x for x in val_vivids if x not in vidids_black_gloves]
        test_vivds = [x for x in test_vivds if x not in vidids_black_gloves]
        

    if filter_blue_gloves:
        train_vidids = [x for x in train_vidids if x not in vidids_blue_gloves]
        val_vivids = [x for x in val_vivids if x not in vidids_blue_gloves]
        test_vivds = [x for x in test_vivds if x not in vidids_blue_gloves]
        
    
    ## individual splits by gids
    # total = len(dset.index.imgs)
    # inds = [i for i in range(1, total+1)]
    # train_size, val_size = int(0.8*total), int(0.1*total)
    # test_size = total - train_size - val_size
    # train_inds = set(list(np.random.choice(inds, size=train_size, replace=False)))
    # remaining_inds = list(set(inds) - train_inds)
    # val_inds = set(list(np.random.choice(remaining_inds, size=val_size, replace=False)))
    # test_inds = list(set(remaining_inds) - val_inds)
    # # test_inds = list(np.random.choice(remaining_inds, size=test_size, replace=False))

    # train_img_ids = [dset.index.imgs[i]['id'] for i in train_inds]
    # val_img_ids = [dset.index.imgs[i]['id'] for i in val_inds]
    # test_img_ids = [dset.index.imgs[i]['id'] for i in test_inds]
    # print(f"train: {len(train_inds)}, val: {len(val_inds)}, test: {len(test_inds)}")

    for vid in train_vidids:
        # print(type(dset.index.vidid_to_gids[vid]))
        # print(type(list(dset.index.vidid_to_gids[vid])))
        # print(list(dset.index.vidid_to_gids[vid]))
        if vid in dset.index.vidid_to_gids.keys():
            train_img_ids.extend(list(dset.index.vidid_to_gids[vid]))

    for vid in val_vivids:
        
        if vid in dset.index.vidid_to_gids.keys():
            val_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
        # val_img_ids = set(val_img_ids) + set(dset.index.vidid_to_gids[vid])

    for vid in test_vivds:
        
        if vid in dset.index.vidid_to_gids.keys():
            test_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
        # test_img_ids = set(test_img_ids) + set(dset.index.vidid_to_gids[vid])



    train_img_ids, val_img_ids, test_img_ids = list(train_img_ids), list(val_img_ids), list(test_img_ids)

    print(f"train num_images: {len(train_img_ids)}, val num_images: {len(val_img_ids)}, test num_images: {len(test_img_ids)}")

    train_dset = dset.subset(gids=train_img_ids, copy=True)
    val_dset = dset.subset(gids=val_img_ids, copy=True)
    test_dset = dset.subset(gids=test_img_ids, copy=True)
# exit()


# for split in ["train_activity", "val", "test"]:
for dset, split in zip([train_dset, val_dset, test_dset], ["train_activity", "val", "test"]):
    # kwcoco_file = f"{obj_dets_dir}/{obj_exp_name}_results_{split}_conf_0.1_plus_hl_hands_new_obj_labels.mscoco.json"

    # print(f"dset length: {len(dset.index.imgs)}")
    # exit()
    # print(f"kps cats: {dset.keypoint_categories()}")
    org_num_vidoes = len(list(dset.index.videos.keys()))
    new_num_videos = 0
    for video_id in ub.ProgIter(dset.index.videos.keys()):
        
        # if video_id != 10:
        #     continue
        
        video = dset.index.videos[video_id]
        video_name = video["name"]
        if "_extracted" in video_name:
            video_name = video_name.split("_extracted")[0]

        image_ids = dset.index.vidid_to_gids[video_id]
        num_images = len(image_ids)
        
        print(f"there are {num_images} in video {video_id}")

        video_dset = dset.subset(gids=image_ids, copy=True)

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
                objects_joints=feat_to_bools[feat_type][0],
                hands_joints=feat_to_bools[feat_type][1],
                aug_trans_range = aug_trans_range,
                aug_rot_range = aug_rot_range,
                top_n_objects=2
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
                for image_id in image_ids:
                    image = dset.imgs[image_id]
                    image_n = image["file_name"] # this is the shortened string

                    frame_idx, time = time_from_name(image_n)
                    
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
