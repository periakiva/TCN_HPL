import torch
from mmpose.apis import inference_top_down_pose_model
import numpy as np
from mmpose.datasets import DatasetInfo
import warnings


def predict_single(det_model, pose_model, image: torch.tensor, bbox_thr=None) -> list:

    keypoints_cats = [
        "nose",
        "mouth",
        "throat",
        "chest",
        "stomach",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "left_upper_leg",
        "right_upper_leg",
        "left_knee",
        "right_knee",
        "left_lower_leg",
        "right_lower_leg",
        "left_foot",
        "right_foot",
        "back",
    ]

    keypoints_cats_dset = [
        {"name": value, "id": index} for index, value in enumerate(keypoints_cats)
    ]

    pose_dataset = pose_model.cfg.data["test"]["type"]
    pose_dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)

    pose_dataset_info = pose_model.cfg.data["test"].get("dataset_info", None)
    if pose_dataset_info is None:
        warnings.warn(
            "Please set `dataset_info` in the config."
            "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
            DeprecationWarning,
        )
    else:
        pose_dataset_info = DatasetInfo(pose_dataset_info)

    predictions, _ = det_model.run_on_image(image)
    instances = predictions["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes.tolist() if instances.has("pred_classes") else None

    boxes_list, labels_list, keypoints_list = [], [], []

    if boxes is not None:

        # person_results = []
        for box_id, _bbox in enumerate(boxes):

            box_class = classes[box_id]
            if box_class == 0:
                pred_class = box_class
                pred_label = "patient"
            elif box_class == 1:
                pred_class = box_class
                pred_label = "user"

            boxes_list.append(np.asarray(_bbox).tolist())
            labels_list.append(pred_label)

            current_ann = {}
            # current_ann['id'] = ann_id
            current_ann["image_id"] = 0
            current_ann["bbox"] = np.asarray(_bbox).tolist()  # _bbox
            current_ann["category_id"] = pred_class
            current_ann["label"] = pred_label
            current_ann["bbox_score"] = f"{scores[box_id] * 100:0.2f}"

            if box_class == 0 and float(current_ann["bbox_score"]) > bbox_thr:
                person_results = [current_ann]

                pose_results, returned_outputs = inference_top_down_pose_model(
                    model=pose_model,
                    img_or_path=image,
                    person_results=person_results,
                    bbox_thr=None,
                    format="xyxy",
                    dataset=pose_dataset,
                    dataset_info=pose_dataset_info,
                    return_heatmap=False,
                    outputs=["backbone"],
                )

                pose_keypoints = pose_results[0]["keypoints"].tolist()
                keypoints_list.append(pose_keypoints)

    return boxes_list, labels_list, keypoints_list
