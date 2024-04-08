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

LAB_TASK_TO_NAME = {
    'm2': "M2_Lab_Skills",
    'm3': "M3_Lab_Skills",
    'm5': "M5_Lab_Skills",
    'r18': "R18_Lab_Skills",
}

FEAT_TO_BOOL = {
        "no_pose": [False, False],
        "with_pose": [True, True],
        "only_hands_joints": [False, True],
        "only_objects_joints": [True, False]
    }


def main(task: str, ptg_root: str, config_root: str, data_type: str):
    config_path = f"{config_root}/experiment/{task}/feat_v6.yaml"
    config = load_yaml_as_dict(config_path)

    activity_config_path = f"{ptg_root}/config/activity_labels/medical"
    feat_version = 6

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

    print(f"Generating features for task: {task}")
    
    if data_type == "gyges":
        dset = kwcoco.CocoDataset(config['paths']['dataset_kwcoco'])
    elif data_type == "bbn":
        dset = kwcoco.CocoDataset(config['paths']['dataset_kwcoco_lab'])

    if reshuffle_datasets:
        
        train_img_ids, val_img_ids, test_img_ids = [], [], []
        if data_type == "gyges":
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
        
        elif data_type == "bbn":
            task_name = config['task'].upper()
            print(f"task name: {task_name}")
            print(f"dset.index.name_to_video.keys(): {dset.index.name_to_video}")
            all_vids = sorted(list(dset.index.name_to_video.keys()))
            print(f"all vids: {all_vids}")
            train_vidids = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['train_vid_ids_bbn']]
            val_vivids = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['val_vid_ids_bbn']]
            test_vivds = [dset.index.name_to_video[vid_name]['id'] for vid_name in all_vids if dset.index.name_to_video[vid_name]['id'] in config['data_gen']['test_vid_ids_bbn']]
            # val_vivids = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['val_vid_ids_bbn'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
            # test_vivds = [dset.index.name_to_video[f"{task_name}-{index}"]['id'] for index in config['data_gen']['test_vid_ids_bbn'] if f"{task_name}-{index}" in dset.index.name_to_video.keys()]
            
        
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

        train_img_ids, val_img_ids, test_img_ids = list(train_img_ids), list(val_img_ids), list(test_img_ids)

        print(f"train num_images: {len(train_img_ids)}, val num_images: {len(val_img_ids)}, test num_images: {len(test_img_ids)}")

        train_dset = dset.subset(gids=train_img_ids, copy=True)
        val_dset = dset.subset(gids=val_img_ids, copy=True)
        test_dset = dset.subset(gids=test_img_ids, copy=True)

    if data_type == "gyges":
        skill_data_root = f"{config['paths']['bbn_data_dir']}/Release_v0.5/v0.56/{TASK_TO_NAME[task]}/Data"
    elif data_type == "bbn":
        skill_data_root = f"{config['paths']['bbn_data_dir']}/lab_data/{LAB_TASK_TO_NAME[task]}"
    # videos_names = os.listdir(skill_data_root)
    # for split in ["train_activity", "val", "test"]:
    for dset, split in zip([train_dset, val_dset, test_dset], ["train_activity", "val", "test"]):
        # org_num_vidoes = len(list(dset.index.videos.keys()))
        # new_num_videos = 0
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
            if data_type == "gyges":
                video_root = f"{skill_data_root}/{video_name}"
                activity_gt_file = f"{video_root}/{video_name}.skill_labels_by_frame.txt"
            elif data_type == "bbn":
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
            
            
            # print(f"str type: {text}")
            
            activityi_gt_list = ["background" for x in range(num_images)]
            # print(f"activity_gt_file: {activity_gt_file}")
            # print(f"activity_labels_desc_mapping: {activity_labels_desc_mapping}")
            # print(f"text_list: {text_list}")
            if data_type == "gtges":
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
                        # print(f"label_index: {label_index}")
                        activityi_gt_list[label_index] = gt_label
                    
            elif data_type == "bbn":
                text = text.replace('\n', '\t')
                text_list = text.split("\t")#[:-1]
                # print(f"text_list: {text_list}")
                for index in range(0, len(text_list), 4):
                    triplet = text_list[index:index+4]
                    # print(f"index: {index}, {text_list[index]}")
                    # print(f"triplet: {text_list[index:index+4]}")
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
        default="gyges",
        help="gyges=proferssional data, bbn=use lab data",
        type=str,
    )
    
    args = parser.parse_args()
    main(task=args.task, ptg_root=args.ptg_root, config_root=args.config_root, data_type=args.data_type)