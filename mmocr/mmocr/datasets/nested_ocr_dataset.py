import os
import json
from mmocr.registry import DATASETS  # Ensure you import the correct registry
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to display info logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Output logs to console
)

log = logging.getLogger(__name__)  # Use __name__ to identify the logger


@DATASETS.register_module()
class NestedOCRDataset:
    def __init__(self, root_dir, checkpoint_path='/vast/zw4603/mmocr/converted_dataset/MMOCR_format_dataset.json'):
        self.image_paths = []
        self.annotations = []
        self.data = ""

        if os.path.exists(checkpoint_path):
            log.info("Loading dataset from checkpoint...")
            self.load_checkpoint(checkpoint_path)
        else:
            log.info("Checkpoint not found. Running conversion...")
            self.load_data(root_dir)
            self.save_checkpoint(checkpoint_path)

    def load_data(self, root_dir):
        """Recursively load and convert JSON annotations and corresponding images."""
        image_id_to_file = {}
        # First pass: Collect image file names by image_id
        for root, dirs, _ in os.walk(root_dir):
            for dir in dirs:
                if dir.startswith('SNGS'):
                    sngs_path = os.path.join(root, dir)
                    for filename in os.listdir(sngs_path):
                        file_path = os.path.join(sngs_path, filename)
                        if file_path.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                for img in data.get('images', []):
                                    image_id_to_file[img['image_id']] = img['file_name']

        # Second pass: Collect annotations and valid image paths
        for root, dirs, _ in os.walk(root_dir):
            for dir in dirs:
                if dir.startswith('SNGS'):
                    sngs_path = os.path.join(root, dir)
                    for filename in os.listdir(sngs_path):
                        file_path = os.path.join(sngs_path, filename)
                        if file_path.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)

                                # Filter annotations safely
                                filtered_annotations = [
                                    ann for ann in data.get('annotations', [])
                                    if ann.get("attributes", {}).get("role") == "player"
                                    and ann["attributes"].get("jersey") is not None
                                ]

                                self.annotations.extend(filtered_annotations)

                                # Collect image paths for valid annotations
                                for ann in filtered_annotations:
                                    image_id = ann['image_id']
                                    if image_id in image_id_to_file:
                                        file_name = image_id_to_file[image_id]
                                        image_path = os.path.join(root, "img1", file_name)
                                        self.image_paths.append(image_path)
                                    else:
                                        log.warning(f"Image ID {image_id} not found.")

    def save_checkpoint(self, checkpoint_path):
        """Save the dataset to a checkpoint file."""
        checkpoint = {
            "annotations": self.annotations,
            "image_paths": self.image_paths
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)
        log.info(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load the dataset from a checkpoint file."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.annotations = checkpoint["annotations"]
        self.image_paths = checkpoint["image_paths"]
        log.info("Dataset loaded from checkpoint.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return image path and corresponding annotation."""
        return self.image_paths[idx], self.annotations[idx]
