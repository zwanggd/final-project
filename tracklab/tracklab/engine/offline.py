import logging

from tracklab.engine import TrackingEngine
from tracklab.utils.cv2 import cv2_load_image
import subprocess
from tracklab.engine.clean_detection import clean_detection_csv
from tracklab.engine.detection_to_tracklet import convert_dataset
from tracklab.engine.create_detection_with_jersey import add_jersey_numbers_to_csv

log = logging.getLogger(__name__)
import pandas as pd
import numpy as np

# bbox_csv_path = '/Users/kai/GSR/soccernet/model_detections/BPBReIDStrongSORT_detection.csv'
bbox_csv_path = '/Users/kai/GSR/soccernet/model_detections/NBJW_Calib_detection.csv'
role_csv_path = '/Users/kai/GSR/soccernet/model_detections/PRTReId_detection.csv'
cleaned_csv_path = '/Users/kai/GSR/soccernet/model_detections/NBJW_Calib_detection_cleaned.csv'
base_dir = '/Users/kai/GSR/data/SoccerNetGS'
output_dir = '/Users/kai/GSR/data/SoccerNetGS/converted'

jersey_number_json_path = '/Users/kai/GSR/soccernet/jersey-number-pipeline/out/SoccerNetResults/challenge_final_results.json'
output_csv_path = '/Users/kai/GSR/soccernet/model_detections/NBJW_Calib_detection_cleaned_with_jersey.csv'


class OfflineTrackingEngine(TrackingEngine):
    def video_loop(self, tracker_state, video, video_id):
        for name, model in self.models.items():
            if hasattr(model, "reset"):
                model.reset()

        detections, image_pred = tracker_state.load()
        if len(self.module_names) == 0:
            return detections, image_pred
        image_filepaths = {idx: fn for idx, fn in image_pred["file_path"].items()}
        model_names = self.module_names
        log.info(f"Offlinetrack, model names: {model_names}")
        NBJW_Calib_detection = []
        PRTReId_detection = []
        NBJW_Calib_detection_cleaned = []
        NBJW_Calib_detection_cleaned_with_jersey = []

        for model_name in model_names:
            log.info(f"model: {model_name}, detection: {detections}")\
                
            if(model_name == "MMOCR"):
                log.info("Skipping MMOCR")
                # NBJW_Calib_detection_cleaned = clean_detection_csv(NBJW_Calib_detection, PRTReId_detection)
                convert_dataset(NBJW_Calib_detection, base_dir, output_dir )
                try:
                    result = subprocess.run(['python', '/Users/kai/GSR/soccernet/jersey-number-pipeline/main.py', 'SoccerNet', 'challenge'], check=True)
                    log.info("The second main.py script completed successfully.")
                    NBJW_Calib_detection_cleaned_with_jersey = add_jersey_numbers_to_csv(NBJW_Calib_detection, jersey_number_json_path)
                except subprocess.CalledProcessError as e:
                    print(f"The second main.py script failed with return code {e.returncode}")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                continue
            if self.models[model_name].level == "video":
                if(model_name == "MajorityVoteTracklet"):
                    detections = NBJW_Calib_detection_cleaned_with_jersey
                    log.info(f"MajorityVoteTracklet: {detections}")

                detections = self.models[model_name].process(detections, image_pred)
                continue
            self.datapipes[model_name].update(image_filepaths, image_pred, detections)
            self.callback(
                "on_module_start",
                task=model_name,
                dataloader=self.dataloaders[model_name],
            )
            for batch in self.dataloaders[model_name]:
                detections, image_pred = self.default_step(batch, model_name, detections, image_pred)
            self.callback("on_module_end", task=model_name, detections=detections)
            if detections.empty:
                return detections, image_pred
            
            if(model_name == "NBJW_Calib"):
                NBJW_Calib_detection = detections
            if(model_name == "PRTReId"):
                PRTReId_detection = detections
            # if 'embeddings' in detections.columns:
            #     d = detections
            #     d['embeddings'] = d['embeddings'].apply(lambda x: np.array2string(x, separator=','))
            #     d.to_csv(f"/Users/kai/GSR/soccernet/model_detections/{model_name}_detection.csv")
            #     continue
            # detections.to_csv(f"/Users/kai/GSR/soccernet/model_detections/{model_name}_detection.csv")
        return detections, image_pred
