import torch
import numpy as np
import pandas as pd
from pathlib import Path

from tracklab.pipeline import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh
import strong_sort.strong_sort as strong_sort

import logging

from tracklab.utils.cv2 import cv2_load_image

log = logging.getLogger(__name__)

@staticmethod
def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_, y1_, x2_, y2_ = bbox2
    inter_x1, inter_y1 = max(x1, x1_), max(y1, y1_)
    inter_x2, inter_y2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y1 - inter_y2)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_ - x1_) * (y2_ - y1_)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0


class StrongSORT(ImageLevelModule):
    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()
        self.cross_data = {}  # 用于存储交叉信息
        self.frame_counter = 0  # 记录当前帧数

    input_columns = [
        "bbox_ltwh",
        "bbox_conf",
        "category_id",
    ]
    output_columns = ["track_id", "track_bbox_ltwh", "track_bbox_conf"]

    def __init__(self, cfg, device, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(
            Path(self.cfg.model_weights),
            self.device,
            self.cfg.fp16,
            **self.cfg.hyperparams
        )
        # For camera compensation
        self.prev_frame = None

    def save_cross_info(self, track1, track2, bbox1, bbox2):
        self.cross_data[(track1, track2)] = {
            "track1_bbox": bbox1,
            "track2_bbox": bbox2,
            "last_frame": self.frame_counter,
        }


    def recover_tracks_if_separated(self, results):
        to_delete = []
        for (track1, track2), info in self.cross_data.items():
            iou = self.calculate_iou(info["track1_bbox"], info["track2_bbox"])
            if iou < self.cfg.cross_exit_threshold:
                log.info(f"Tracks {track1} and {track2} have separated.")
                self.assign_historical_info(track1, info["track1_bbox"])
                self.assign_historical_info(track2, info["track2_bbox"])
                to_delete.append((track1, track2))
        for key in to_delete:
            del self.cross_data[key]

    def handle_crossing_tracks(self, track1, track2):
        if (track1, track2) not in self.cross_data:
            self.save_cross_info(track1, track2, self.cross_data[track1]["bbox"], self.cross_data[track2]["bbox"])
        log.info(f"Handling cross for {track1} and {track2}")


    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):
        processed_detections = []
        if len(detections) == 0:
            return {"input": []}
        for det_id, detection in detections.iterrows():
            ltrb = detection.bbox.ltrb()
            conf = detection.bbox.conf()
            cls = detection.category_id
            tracklab_id = int(detection.name)

            for other_det_id, other_detection in detections.iterrows():
                if det_id == other_det_id:
                    continue
                other_ltrb = other_detection.bbox.ltrb()
                iou = calculate_iou(ltrb, other_ltrb)
                if iou > self.cfg.cross_threshold:  # 配置中定义交叉阈值
                    log.info(f"Cross detected between {det_id} and {other_det_id}")
                    save_cross_info(det_id, other_det_id, ltrb, other_ltrb)

            processed_detections.append(
                np.array([*ltrb, conf, cls, tracklab_id])
            )
        return {
            "input": np.stack(processed_detections)
        }

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        self.frame_counter += 1
        
        image = cv2_load_image(metadatas['file_path'].values[0])
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.model.tracker.camera_update(self.prev_frame, image)
            self.prev_frame = image
        if len(detections) == 0:
            return []
        inputs = batch["input"][0]  # Nx7 [l,t,r,b,conf,class,tracklab_id]
        inputs = inputs[inputs[:, 4] > self.cfg.min_confidence]
        results = self.model.update(inputs, image)

        for track in results:
            track_id = track[4]
            bbox = track[:4]
            for other_track in results:
                other_track_id = other_track[4]
                if track_id == other_track_id:
                    continue
                other_bbox = other_track[:4]
                iou = calculate_iou(bbox, other_bbox)
                if iou > self.cfg.cross_threshold:
                    log.info(f"Tracks {track_id} and {other_track_id} are crossing.")
                    handle_crossing_tracks(track_id, other_track_id)

        recover_tracks_if_separated(results)

        results = np.asarray(results)  # N'x9 [l,t,r,b,track_id,class,conf,queue,idx]
        if results.size:
            track_bbox_ltwh = [ltrb_to_ltwh(x) for x in results[:, :4]]
            track_bbox_conf = list(results[:, 6])
            track_ids = list(results[:, 4])
            idxs = list(results[:, 8].astype(int))
            # FIXME should be a subset but sometimes returns an idx that was in the previous
            # batch of detections... For the moment, we let the override happen
            # assert set(idxs).issubset(
            #    detections.index
            # ), "Mismatch of indexes during the tracking. The results should match the detections."
            results = pd.DataFrame(
                {
                    "track_bbox_ltwh": track_bbox_ltwh,
                    "track_bbox_conf": track_bbox_conf,
                    "track_id": track_ids,
                    "idxs": idxs,
                }
            )
            results.set_index("idxs", inplace=True, drop=True)
            return results
        else:
            return []
