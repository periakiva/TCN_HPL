import os
import yaml
import glob
import warnings
import kwcoco

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
feat_type = "with_pose" #[no_pose, with_pose, only_hands_joints, only_objects_joints]
feat_to_bools = {
    "no_pose": [False, False],
    "with_pose": [True, True],
    "only_hands_joints": [False, True],
    "only_objects_joints": [True, False]
}
exp_name = f"p_m2_tqt_data_test_feat_v6_{feat_type}" #[p_m2_tqt_data_test_feat_v6_with_pose, p_m2_tqt_data_test_feat_v6_only_hands_joints, p_m2_tqt_data_test_feat_v6_only_objects_joints, p_m2_tqt_data_test_feat_v6_no_pose]
if using_done:
    exp_name = f"{exp_name}_done_gt"

output_data_dir = f"/data/PTG/TCN_data/m2/{exp_name}"

gt_dir = f"{output_data_dir}/groundTruth"
frames_dir = f"{output_data_dir}/frames"
bundle_dir = f"{output_data_dir}/splits"
features_dir = f"{output_data_dir}/features"

# Create directories
for folder in [output_data_dir, gt_dir, frames_dir, bundle_dir, features_dir]:
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

#####################
# Features,
# groundtruth and
# bundles
#####################
### create splits dsets

kwcoco_file = "/home/local/KHQ/peri.akiva/projects/medical-pose/ViTPose/results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets_with_pose.mscoco.json"
dset = kwcoco.CocoDataset(kwcoco_file)
train_img_ids, val_img_ids, test_img_ids = [], [], []

train_vidids = [1, 7, 13, 19, 21, 30,
            31, 32, 33, 34, 35, 36, 39, 52, 53,
            57, 58, 60, 64, 70, 72, 73, 75] 


# 1 7 13 19 21 26 27 29 30 31 32 33 34 35 36 38 39 40 
# 52 53 57 58 60 63 64 70 71 72 73 74 75 76 77 119 122 124 132 133
# [1, 7, 13, 19, 21, 26, 27, 29, 30,
#             31, 32, 33, 34, 35, 36, 38, 39, 40, 52, 53,
#             57, 58, 60, 63, 64, 70, 71, 72, 73, 74, 75, 
#             76, 77, 119, 122, 124,
#             132, 133] 
# val_vivids = [72, 73, 74, 75]

val_vivids = [19, 77, 74]
# test_vivds= [63, 71, 40]


test_vivds= [19, 77, 74]

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

    train_img_ids.extend(list(dset.index.vidid_to_gids[vid]))

for vid in val_vivids:
    val_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
    # val_img_ids = set(val_img_ids) + set(dset.index.vidid_to_gids[vid])

for vid in test_vivds:
    test_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
    # test_img_ids = set(test_img_ids) + set(dset.index.vidid_to_gids[vid])

train_img_ids, val_img_ids, test_img_ids = list(train_img_ids), list(val_img_ids), list(test_img_ids)

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
        X, y = compute_feats(
            act_map,
            image_activity_gt,
            image_id_to_dataset,
            label_to_ind,
            act_id_to_str,
            ann_by_image,
            feat_version=feat_version,
            objects_joints=feat_to_bools[feat_type][0],
            hands_joints=feat_to_bools[feat_type][1]
        )

        X = X.T
        print(f"X after transpose: {X.shape}")

        np.save(f"{features_dir}/{video_name}.npy", X)

        # groundtruth
        with open(f"{gt_dir}/{video_name}.txt", "w") as gt_f, \
            open(f"{frames_dir}/{video_name}.txt", "w") as frames_f:
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
            bundle.write(f"{video_name}.txt\n")

print("Done!")
print(f"Saved training data to {output_data_dir}")
