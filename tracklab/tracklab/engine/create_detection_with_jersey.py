import os
import pandas as pd
import json
import math
import numpy as np

def add_jersey_numbers_to_csv(NBJW_Calib_detection_cleaned, jersey_number_json_path):
    df = NBJW_Calib_detection_cleaned

    print(f"add_jersey_numbers_to_csv: {df}")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"Loaded DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")

    print(f"Loading jersey number results from: {jersey_number_json_path}")
    with open(jersey_number_json_path, 'r') as f:
        jersey_number_results = json.load(f)

    if 'imgs' in jersey_number_results:
        del jersey_number_results['imgs']

    jersey_number_results = {
            int(k): int(v) if v != -1 else -1
            for k, v in jersey_number_results.items()
        }
    df['tracklet'] = df['track_id']
    df['jersey_number_detection'] = df['tracklet'].map(jersey_number_results)

    for i in range(len(df["jersey_number_detection"])):
            if math.isnan(df["jersey_number_detection"][i]):
                df.loc[i, "jersey_number_confidence"] = 0.0
            elif df["jersey_number_detection"][i] == -1.0:
                df.loc[i, "jersey_number_confidence"] = 0.0
                df.loc[i, "jersey_number_detection"] = float('nan')
            else:
                df.loc[i, "jersey_number_confidence"] = 1.0

    df = df.drop(columns=['tracklet'])
    df.to_csv("/Users/kai/GSR/soccernet/debug.csv")
    return df


if __name__ == "__main__":
    # Example usage
    cleaned_csv_path = '/Users/kai/GSR/soccernet/model_detections/BPBReIDStrongSORT_detection_cleaned.csv'
    jersey_number_json_path = '/Users/kai/GSR/soccernet/jersey-number-pipeline/out/SoccerNetResults/challenge_final_results.json'
    output_csv_path = '/Users/kai/GSR/soccernet/model_detections/BPBReIDStrongSORT_detection_cleaned_with_jersey.csv'

    add_jersey_numbers_to_csv(cleaned_csv_path, jersey_number_json_path, output_csv_path)
