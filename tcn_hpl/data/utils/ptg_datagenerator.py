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
obj_exp_name = "p_bbn_model_m2_m3_m5_r18_v10" #"coffee+tea_yolov7"


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
exp_name = "p_m2_tqt_data_test_feat_v6"#f"coffee_and_tea_feat_v{str(feat_version)}"
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
for split in ["train_activity", "val", "test"]:
    # kwcoco_file = f"{obj_dets_dir}/{obj_exp_name}_results_{split}_conf_0.1_plus_hl_hands_new_obj_labels.mscoco.json"
    kwcoco_file = "/home/local/KHQ/peri.akiva/projects/medical-pose/ViTPose/results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets_with_pose.mscoco.json"
    dset = kwcoco.CocoDataset(kwcoco_file)

    # print(f"kps cats: {dset.keypoint_categories()}")
    
    for video_id in ub.ProgIter(
        dset.index.videos.keys(), desc=f"Creating features for videos in {split}"
    ):
        
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
            feat_version=feat_version
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
