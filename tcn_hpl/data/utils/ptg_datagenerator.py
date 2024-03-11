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
# from angel_system.data.data_paths import grab_data, data_dir

def load_yaml_as_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    return config_dict

#####################
# Inputs
#####################

TASK_TO_NAME = {
    'm1': "M1_Trauma_Assessment",
    'm2': "M2_Tourniquet",
    'm3': "M3_Pressure_Dressing",
    'm4': "M4_Wound_Packing",
    'm5': "M5_X-Stat",
    'r18': "R18_Chest_Seal",
}

task = "m3"
config_path = f"/home/local/KHQ/peri.akiva/projects/TCN_HPL/configs/experiment/{task}/feat_v6.yaml"
config = load_yaml_as_dict(config_path)



print(f"config: {config}")
# exit()
obj_exp_name = "p_bbn_model_m2_m3_m5_r18_v11" #"coffee+tea_yolov7"


#  f"{ptg_root}/config/activity_labels/medical"
# obj_dets_dir = f"{data_dir}/annotations/{recipe}/results/{obj_exp_name}"
# obj_dets_dir = "/data/PTG/cooking/object_anns/old_coffee/results/coffee_base/" #"/home/local/KHQ/hannah.defazio/yolov7/runs/detect/coffee+tea_yolov7/"
# obj_dets_dir = "/home/local/KHQ/peri.akiva/projects/medical-pose/bbox_detection_results/RESULTS_m2_with_lab_cleaned_fixed_data_with_steps_results_train_activity_with_patient_dets.mscoco.json"
ptg_root = "/home/local/KHQ/peri.akiva/angel_system"
activity_config_path = f"{ptg_root}/config/activity_labels/medical"
# activity_config_fn = f"{activity_config_path}/task_{task}.yaml"
# activity_config_fn = "/home/local/KHQ/peri.akiva/projects/angel_system/config/activity_labels/medical_tourniquet.v3.yaml"

feat_version = 6
using_done = False # Set the gt according to when an activity is done

#####################
# Output
#####################
# reshuffle_datasets = True
# augment = False
# num_augs = 5

reshuffle_datasets, augment, num_augs = config['data_gen']['reshuffle_datasets'], config['data_gen']['augment'], config['data_gen']['num_augs']

if augment:
    num_augs = num_augs
    aug_trans_range, aug_rot_range = [-5, 5], [-5, 5]
else:
    num_augs = 1
    aug_trans_range, aug_rot_range = None, None

# feat_type = "with_pose" #[no_pose, with_pose, only_hands_joints, only_objects_joints]
feat_type = config['data_gen']['feat_type']
feat_to_bools = {
    "no_pose": [False, False],
    "with_pose": [True, True],
    "only_hands_joints": [False, True],
    "only_objects_joints": [True, False]
}
exp_name = f"p_{config['task']}_feat_v6_{feat_type}_v3_aug_{augment}_reshuffle_{reshuffle_datasets}" #[p_m2_tqt_data_test_feat_v6_with_pose, p_m2_tqt_data_test_feat_v6_only_hands_joints, p_m2_tqt_data_test_feat_v6_only_objects_joints, p_m2_tqt_data_test_feat_v6_no_pose]

if using_done:
    exp_name = f"{exp_name}_done_gt"

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
        activity_labels_desc_mapping[label['description']] = label["label"]
        if label_str == "done":
            continue
        mapping.write(f"{i} {label_str}\n")

###################
# Features,
# groundtruth and
# bundles
#####################
### create splits dsets

# names_black_gloves = [22,23,26,24,25,27,29,28,41,42,43,44,45,46,47,48,49,78,
#                        79,84,88,90,80,81,82,83,85,86,87,89,91,99,110,111,121,113,115,116]
# names_blue_gloves = [132,133,50,51,54,55,56,52,61,59,53,57,62,65,66,67,68,69,
#                       58,60,63,64,125,126,127,129,131,134,135,136,128,130,137,
#                       138,139]


# train_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_train_activity_obj_results_with_dets_and_pose.mscoco.json"
# val_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_val_obj_results_with_dets_and_pose.mscoco.json"
# test_file = "/data/users/peri.akiva/datasets/ptg/m2_good_images_only_no_amputation_stump_test_obj_results_with_dets_and_pose.mscoco.json"

# train_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_train_activity_obj_results.mscoco.json"
# val_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_val_obj_results.mscoco.json"
# test_file = "/data/PTG/medical/training/yolo_object_detector/detect/m2_good_images_only_no_amputation_stump/m2_good_images_only_no_amputation_stump_test_obj_results.mscoco.json"
# train_dset = kwcoco.CocoDataset(train_file)
# val_dset = kwcoco.CocoDataset(val_file)
# test_dset = kwcoco.CocoDataset(test_file)

dset = kwcoco.CocoDataset(config['paths']['dataset_kwcoco'])

if reshuffle_datasets:
    # dset = kwcoco.CocoDataset.union(train_dset,val_dset,test_dset)
    
    # print(f"vids: {dset.index.videos}")
    
    train_img_ids, val_img_ids, test_img_ids = [], [], []

    # train_names = [1, 2, 4, 8, 9, 10, 11, 12, 16, 17,18, 20, 19, 30, 31, 32, 33, 34,35,36,
    #                 7,132,133,50,51,54,56,52,61,53,57,65,66,67,68,69,58,60,63,64,125,126,
    #                 127,129,131,134,135,136,119,122,124,70,72,92,93,94,95,97,98,100,
    #                 101,102,103,104,105,107,108,112,114,117,118,73]
    
    # bad step distribution GT: [37, 6, 76, 39, 38, 30]
    # val_names= [5, 59,106,130,138, 77, 123, 71]
    
    # test_vivds= [5,6,37,59,106,130,138]
    
    # test_names = [3,14,55,62,96,109,128,137,139, 120, 75, 21, 13]
    
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
        else:
            print(f"{vid} not in the train dataset")

    for vid in val_vivids:
        
        if vid in dset.index.vidid_to_gids.keys():
            val_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
        else:
            print(f"{vid} not in the val dataset")
        # val_img_ids = set(val_img_ids) + set(dset.index.vidid_to_gids[vid])

    for vid in test_vivds:
        
        if vid in dset.index.vidid_to_gids.keys():
            test_img_ids.extend(list(dset.index.vidid_to_gids[vid]))
        else:
            print(f"{vid} not in the test dataset")
        # test_img_ids = set(test_img_ids) + set(dset.index.vidid_to_gids[vid])



    train_img_ids, val_img_ids, test_img_ids = list(train_img_ids), list(val_img_ids), list(test_img_ids)

    print(f"train num_images: {len(train_img_ids)}, val num_images: {len(val_img_ids)}, test num_images: {len(test_img_ids)}")

    train_dset = dset.subset(gids=train_img_ids, copy=True)
    val_dset = dset.subset(gids=val_img_ids, copy=True)
    test_dset = dset.subset(gids=test_img_ids, copy=True)
# exit()

skill_data_root = f"{config['paths']['bbn_data_dir']}/Release_v0.5/v0.56/{TASK_TO_NAME[task]}/Data"
videos_names = os.listdir(skill_data_root)
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
        
        #### begin activity GT section
        video_root = f"{skill_data_root}/{video_name}"
        activity_gt_file = f"{video_root}/{video_name}.skill_labels_by_frame.txt"
        
        if not os.path.exists(activity_gt_file):
            continue
        
        f = open(activity_gt_file, "r")
        text = f.read()
        f.close()
        
        # print(f"str type: {type(text)}")
        text = text.replace('\n', '\t')
        text_list = text.split("\t")[:-1]
        
        # print(f"activity_gt_file: {activity_gt_file}")
        # print(f"video_dset: {video_dset.index.imgs}")
        # print(f"max frame index: {max([int(video_dset.index.imgs[id]['frame_index']) for id in video_dset.index.imgs.keys()])}")
        # print(f"len(text_list): {len(text_list)}")
        activityi_gt_list = ["background" for x in range(num_images)]
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
        
        # print(video_dset.index.imgs)
        # print(video_dset.imgs[1])
        # print(video_dset.imgs[2])
        
        # print(f"\n acitivty gt text: {text_list}")
        # print(f"video name: {video_name}")
        # print(f"activity_gt_file: {activity_gt_file}")
        # print(f"videos in data root: {videos_names}")
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
                objects_joints=feat_to_bools[feat_type][0],
                hands_joints=feat_to_bools[feat_type][1],
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
