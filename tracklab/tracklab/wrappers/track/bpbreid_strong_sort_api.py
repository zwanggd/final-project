from collections import defaultdict
from tracklab.data.global_bbox_store import GlobalBboxStore

import torch
import numpy as np
import pandas as pd
import bpbreid_strong_sort.strong_sort as strong_sort
import logging

from tracklab.pipeline import ImageLevelModule

log = logging.getLogger(__name__)


class BPBReIDStrongSORT(ImageLevelModule):
    def __init__(self, cfg, device, batch_size=None, **kwargs):
        super().__init__(batch_size=1)
        self.cfg = cfg
        self.device = device
        self.reset()

    input_columns = [
        "bbox_ltwh",
        "embeddings",
        "visibility_scores",
    ]
    output_columns = [
        "track_id",
        "track_bbox_kf_ltwh",
        "track_bbox_pred_kf_ltwh",
        "matched_with",
        "costs",
        "hits",
        "age",
        "time_since_update",
        "state",
    ]

    def reset(self):
        """Reset the tracker state to start tracking in a new video."""
        self.model = strong_sort.StrongSORT(
            ema_alpha=self.cfg.ema_alpha,
            mc_lambda=self.cfg.mc_lambda,
            max_dist=self.cfg.max_dist,
            motion_criterium=self.cfg.motion_criterium,
            max_iou_distance=self.cfg.max_iou_distance,
            max_oks_distance=self.cfg.max_oks_distance,
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            nn_budget=self.cfg.nn_budget,
            min_bbox_confidence=self.cfg.min_bbox_confidence,
            only_position_for_kf_gating=self.cfg.only_position_for_kf_gating,
            max_kalman_prediction_without_update=self.cfg.max_kalman_prediction_without_update,
            matching_strategy=self.cfg.matching_strategy,
            gating_thres_factor=self.cfg.gating_thres_factor,
            w_kfgd=self.cfg.w_kfgd,
            w_reid=self.cfg.w_reid,
            w_st=self.cfg.w_st,
        )
        # For camera compensation
        self.prev_frame = None

    def prepare_next_frame(self, next_frame: np.ndarray):
        # Propagate the state distribution to the current time step using a Kalman filter prediction step.
        self.model.tracker.predict()

        # Camera motion compensation
        if self.cfg.ecc:
            if self.prev_frame is not None:
                self.model.tracker.camera_update(self.prev_frame, next_frame)
            self.prev_frame = next_frame

    @torch.no_grad()
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series):

        # log.info(f"Detection in preprocess: {detections}")
        if len(detections) == 0:
            return {
            "id": [],
            "bbox_ltwh": [],
            "reid_features": [],
            "visibility_scores": [],
            "scores": [],
            "classes": [],
            "frame": [],
        }
        if hasattr(detections, "bbox_conf"):
            score = detections.bbox.conf()
        else:
            score = detections.keypoints_conf
        input_tuple = {
            "id": detections.index.to_numpy(),
            "bbox_ltwh": np.stack(detections.bbox_ltwh),
            "reid_features": np.stack(detections.embeddings),
            "visibility_scores": np.stack(detections.visibility_scores),
            "scores": np.stack(score),
            "classes": np.zeros(len(detections.index)),
            "frame": np.ones(len(detections.index)) * metadata.frame,
        }
        if "keypoints_xyc" in detections:
            input_tuple["keypoints"] = np.stack(detections.keypoints_xyc)
        return input_tuple

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        if len(detections) == 0:
            return []
        
        GlobalBboxStore.frame_counter += 1
        # Update model with current batch data
        results = self.model.update(
            batch["id"][0],
            batch["bbox_ltwh"][0],
            batch["reid_features"][0],
            batch["visibility_scores"][0],
            batch["scores"][0],
            batch["classes"][0],
            batch["frame"][0],
            batch["keypoints"][0] if "keypoints" in batch else None,
        )
        # detections.to_csv("/Users/kai/GSR/soccernet/detection.csv")

        # Ensure required columns are present
        required_columns = ['track_bbox_kf_ltwh', 'track_id']
        missing_columns = [col for col in required_columns if col not in results.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in results DataFrame: {missing_columns}")

        # Handle crossing tracks
        # for i, track1 in results.iterrows():
        #     for j, track2 in results.iterrows():
        #         if i >= j:
        #             continue
                        
        #         iou = GlobalBboxStore.calculate_iou(track1['track_bbox_kf_ltwh'], track2['track_bbox_kf_ltwh'])
        #         # log.info(f"iou: {iou},  threshold: {self.cfg.cross_threshold}")
        #         if iou > self.cfg.cross_threshold:
        #             log.info(f"Tracks {track1['track_id']} and {track2['track_id']} are crossing.")
        #             result = GlobalBboxStore.handle_crossing_tracks(
        #                 track1['track_id'], track2['track_id'], track1['track_bbox_kf_ltwh'], track2['track_bbox_kf_ltwh'], iou
        #             )
        #             if(result == 0):
        #                 continue
        #             else:
        #                 existing_track, other_track, replace_track = result
        #                 results.loc[results['track_id'] == replace_track, 'track_id'] = other_track
        #         else:
        #             GlobalBboxStore.recover_tracks_if_separated(
        #                 track1['track_id'], track2['track_id'], track1['track_bbox_kf_ltwh'], track2['track_bbox_kf_ltwh'], iou
        #             )

        # Ensure results indexes match detection indexes
        assert set(results.index).issubset(
            detections.index
        ), "Mismatch of indexes during the tracking. The results should match the detections."
        
        # results = []
        return results
