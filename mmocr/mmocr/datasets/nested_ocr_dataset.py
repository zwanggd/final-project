import os
import json
from mmocr.registry import DATASETS  # Ensure you import the correct registry
import logging


log = logging.getLogger(__name__)

@DATASETS.register_module()
class NestedOCRDataset:
    def __init__(self, root_dir):
        self.image_paths = []
        self.annotations = []
        self.load_data(root_dir)

    def load_data(self, root_dir):
        """Recursively load all JSON annotations and corresponding images."""
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_partitions = os.path.join(root, file)
                for partition in file_partitions:
                    if partition.endswith('.json'):
                        ann_file = os.path.join(root, partition)
                        with open(ann_file, 'r') as f:
                            data = json.load(f)
                            log.error("Data:", data)
                            self.annotations.extend(data['annotations'])
                            for ann in data['annotations']:
                                image_id = ann['image_id']
                                image_path = os.path.join(root, f"img1/{image_id}.jpg")
                                self.image_paths.append(image_path)

    def __len__(self):
        # if not self.image_paths:
        #     raise ValueError("The dataset is empty. Check the root directory or annotations.")
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return image path and corresponding annotation."""
        return self.image_paths[idx], self.annotations[idx]
