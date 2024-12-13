import os
import cv2
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import re
import numpy as np

class TrackIDManager:
    def __init__(self):
        self.track_id_registry = {}
        self.next_unique_id = 1

    def get_or_create_unique_track_id(self, original_track_id):
        if original_track_id not in self.track_id_registry:
            self.track_id_registry[original_track_id] = self.next_unique_id
            self.next_unique_id += 1
        return self.track_id_registry[original_track_id]


def extract_player_images(image_path, frame_annotations, tracklet_output_path, track_id_manager):
    """ Extracts player images from the frame based on bbox, track_id, and image_id. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return []

    saved_files = []
    for i, ann in frame_annotations.iterrows():
        # Extract bbox in ltwh format and track ID from the CSV file
        # bbox = re.findall(r"[-+]?\d*\.\d+|\d+", ann['bbox_ltwh'])  # Extracts all numeric values
        bbox = [float(val) for val in ann['bbox_ltwh']]  # Convert to float
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = x1 + int(bbox[2])  # x + width
        y2 = y1 + int(bbox[3])  # y + height
        track_id = ann['track_id']
        
        # Get a unique track ID using our custom manager
        unique_track_id = track_id_manager.get_or_create_unique_track_id(track_id)

        # Ensure bounding box is within image dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        # Crop the player image
        player_img = image[y1:y2, x1:x2]
        
        if player_img.size == 0:
            print(f"Empty crop for image: {image_path}, bbox: ({x1}, {y1}, {x2}, {y2})")
            continue

        # Generate a unique name for the player image
        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        player_output_folder = os.path.join(tracklet_output_path, str(int(track_id)))
        os.makedirs(player_output_folder, exist_ok=True)

        player_filename = f"{int(base_image_name)-1}.jpg"
        player_path = os.path.join(player_output_folder, player_filename)

        # Save the cropped image
        cv2.imwrite(player_path, player_img)
        saved_files.append((player_path, unique_track_id))

    return saved_files


def convert_dataset(NBJW_Calib_detection_cleaned, base_dir, output_dir, split='valid', sequence='SNGS-021'):
    # Directories for validation set
    img1_path = os.path.join(base_dir, split, sequence, 'img1')
    split_output_path = os.path.join(output_dir, split, sequence)

    # Ensure output directory exists
    os.makedirs(split_output_path, exist_ok=True)

    # Initialize track ID manager
    track_id_manager = TrackIDManager()

    df = NBJW_Calib_detection_cleaned
    save = df['image_id']
    df['image_id'] = df['image_id'].astype(str).str[-6:].astype(str)

    # Group the annotations by image_id
    grouped_annotations = df.groupby('image_id')

    all_annotations = []

    # Process each image in the img1 folder
    for img_filename in tqdm(os.listdir(img1_path), desc=f"Processing images in {split}", leave=False):
        if not img_filename.endswith(('.jpg', '.png')):
            continue
        
        # Extract frame id from the image filename (assumes filenames like 000001.jpg)
        image_id = os.path.splitext(img_filename)[0]
        image_path = os.path.join(img1_path, img_filename)

        # Get all annotations corresponding to this image_id
        if image_id not in df['image_id'].values:
            continue
        
        frame_annotations = grouped_annotations.get_group(image_id)

        # Extract player images from the frame
        player_images = extract_player_images(image_path, frame_annotations, split_output_path, track_id_manager)

        # Add player images to the annotation file
        for player_img_path, unique_track_id in player_images:
            relative_path = os.path.relpath(player_img_path, output_dir)
            all_annotations.append(f"{relative_path} {unique_track_id}")

    # Write annotation file
    annotation_file_path = os.path.join(output_dir, f'{split}.txt')
    with open(annotation_file_path, 'w') as f:
        f.writelines(f"{line}\n" for line in all_annotations)

    df['image_id'] = save
    print(f"Dataset conversion completed. Annotations are stored in {output_dir}.")


if __name__ == "__main__":
    # Example usage
    csv_path = '/Users/kai/GSR/soccernet/model_detections/BPBReIDStrongSORT_detection_cleaned.csv'
    base_dir = '/Users/kai/GSR/data/SoccerNetGS'
    output_dir = '/Users/kai/GSR/data/SoccerNetGS/converted'

    convert_dataset(csv_path, base_dir, output_dir, split='valid', sequence='SNGS-021')
