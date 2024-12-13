import pandas as pd
import re

def clean_detection_csv(NBJW_Calib_detection: pd.DataFrame, PRTReId_detection: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the detection DataFrames by removing non-player bounding boxes.
    
    Args:
    NBJW_Calib_detection (pd.DataFrame): DataFrame containing the calibration detection information.
    PRTReId_detection (pd.DataFrame): DataFrame containing the PRTReId detection information.
    
    Returns:
    pd.DataFrame: The cleaned DataFrame with non-player bounding boxes removed.
    """
    # **Step 1: Clean "Unnamed" columns**
    role_df = NBJW_Calib_detection.loc[:, ~NBJW_Calib_detection.columns.str.contains('^Unnamed')]  # Remove "Unnamed" columns
    
    # **Step 2: Copy bbox_ltwh for temporary merge use**
    temp_role_df = role_df.copy()
    temp_role_df['bbox_ltwh'] = temp_role_df['bbox_ltwh'].apply(
        lambda x: [round(float(n), 2) for n in re.findall(r"[-+]?\d*\.\d+|\d+", x.decode('utf-8'))] 
        if isinstance(x, bytes) 
        else [round(float(n), 2) for n in re.findall(r"[-+]?\d*\.\d+|\d+", x)]
    )

    # **Step 3: Filter non-players**
    non_player_bboxes = temp_role_df[role_df['role_detection'] != 'player'][['image_id', 'bbox_ltwh']]
    print(f"Identified {len(non_player_bboxes)} non-player bboxes to be removed.")

    # **Step 4: Clean "Unnamed" columns in PRTReId_detection**
    bbox_df = PRTReId_detection.loc[:, ~PRTReId_detection.columns.str.contains('^Unnamed')]  # Remove "Unnamed" columns

    # **Step 5: Copy bbox_ltwh for temporary merge use**
    temp_bbox_df = bbox_df.copy()
    temp_bbox_df['bbox_ltwh'] = temp_bbox_df['bbox_ltwh'].apply(
        lambda x: [round(float(n), 2) for n in re.findall(r"[-+]?\d*\.\d+|\d+", x.decode('utf-8'))] 
        if isinstance(x, bytes) 
        else [round(float(n), 2) for n in re.findall(r"[-+]?\d*\.\d+|\d+", x)]
    )

    # **Step 6: Identify track_ids to remove**
    merged_df = pd.merge(
        temp_bbox_df[['image_id', 'bbox_ltwh', 'track_id']],
        non_player_bboxes,
        on=['image_id', 'bbox_ltwh'],
        how='inner'
    )

    # Get the list of track_ids to remove
    track_ids_to_remove = merged_df['track_id'].unique()
    print(f"Removing {len(track_ids_to_remove)} track_ids: {track_ids_to_remove}")

    # **Step 7: Remove all track_ids from PRTReId_detection**
    cleaned_bbox_df = bbox_df[~bbox_df['track_id'].isin(track_ids_to_remove)]

    # **Step 8: Remove "Unnamed" columns once more for safety**
    cleaned_bbox_df = cleaned_bbox_df.loc[:, ~cleaned_bbox_df.columns.str.contains('^Unnamed')]

    return cleaned_bbox_df

if __name__ == "__main__":
    # Replace these with actual DataFrame objects instead of paths
    NBJW_Calib_detection = pd.read_csv('/Users/kai/GSR/soccernet/model_detections/NBJW_Calib_detection.csv')
    PRTReId_detection = pd.read_csv('/Users/kai/GSR/soccernet/model_detections/PRTReId_detection.csv')

    # Call the function with DataFrames instead of paths
    cleaned_bbox_df = clean_detection_csv(NBJW_Calib_detection, PRTReId_detection)
    print(cleaned_bbox_df.head())
