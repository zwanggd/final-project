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
    def __init__(self, root_dir, checkpoint_dir='/vast/zw4603/mmocr/converted_dataset'):
        self.image_paths = []
        self.annotations = []

        # Determine the split (train or val) based on the root directory name
        if 'train' in root_dir.lower():
            self.split = 'train'
        elif 'valid' in root_dir.lower() or 'valid' in root_dir.lower():
            self.split = 'valid'
        else:
            raise ValueError(f"Cannot determine split from root_dir: {root_dir}")

        self.checkpoint_path = os.path.join(checkpoint_dir, f'{self.split}_dataset.json')

        if os.path.exists(self.checkpoint_path):
            log.info(f"Loading {self.split} dataset from checkpoint in {self.checkpoint_path}")
            self.load_checkpoint()
        else:
            log.info(f"{self.split.capitalize()} checkpoint not found. Running conversion...")
            self.load_data(root_dir)
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save the dataset to a checkpoint file."""
        checkpoint = {
            "annotations": self.annotations,
            "image_paths": self.image_paths
        }
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)
        log.info(f"{self.split.capitalize()} checkpoint saved at {self.checkpoint_path}")

    def load_checkpoint(self):
        """Load the dataset from a checkpoint file."""
        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.annotations = checkpoint["annotations"]
        self.image_paths = checkpoint["image_paths"]
        log.info(f"{self.split.capitalize()} dataset loaded from checkpoint.")

    def load_data(self, root_dir):
        """Recursively load and convert JSON annotations and corresponding images."""
        image_id_to_file = {}
        image_path_set = set()  # Use a set to track unique image paths

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

        # Second pass: Collect annotations and filter unique image paths
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

                                # Collect image paths and avoid duplicates
                                for ann in filtered_annotations:
                                    image_id = ann['image_id']
                                    if image_id in image_id_to_file:
                                        file_name = image_id_to_file[image_id]
                                        image_path = os.path.join(root, "img1", file_name)

                                        if image_path not in image_path_set:
                                            self.image_paths.append(image_path)
                                            image_path_set.add(image_path)  # Track as seen

                                        self.annotations.append(ann)
                                    else:
                                        log.warning(f"Image ID {image_id} not found.")

        log.info(f"Loaded {len(self.image_paths)} unique images with annotations.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return image path and corresponding annotations."""
        try:
            image_path = self.image_paths[idx]
            annotations = self.annotations[idx]

            # Return a dictionary to match the model’s expected input format
            return {
                'inputs': image_path,
                'annotations': annotations
            }
        except IndexError as e:
            log.error(f"IndexError: {e}. Returning None for idx {idx}")
            return None

