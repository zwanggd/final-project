import os
import json
from mmocr.registry import DATASETS  # Ensure you import the correct registry
import logging
from torchvision import transforms
from PIL import Image
import torch
from mmengine.structures import InstanceData
from mmocr.structures import TextDetDataSample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

log = logging.getLogger(__name__)

@DATASETS.register_module()
class NestedOCRDataset:
    def __init__(self, root_dir, checkpoint_dir='/Users/kai/GSR/mmocr/converted_dataset'):
        self.image_paths = []
        self.annotations = []
        self.metainfos = []

        # Determine the split
        if 'train' in root_dir.lower():
            self.split = 'train'
        elif 'valid' in root_dir.lower():
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

        # Define image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def save_checkpoint(self):
        checkpoint = {
            "annotations": self.annotations,
            "image_paths": self.image_paths
        }
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)
        log.info(f"{self.split.capitalize()} checkpoint saved at {self.checkpoint_path}")

    def load_checkpoint(self):
        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        self.annotations = checkpoint["annotations"]
        self.image_paths = checkpoint["image_paths"]
        log.info(f"{self.split.capitalize()} dataset loaded from checkpoint.")

    def load_data(self, root_dir):
        """Recursively load JSON annotations and images."""
        image_id_to_file = {}
        image_id_to_height = {}
        image_id_to_width = {}
        image_path_set = set()

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
                                    image_id_to_height[img['image_id']] = img['height']
                                    image_id_to_width[img['image_id']] = img['width']

        for root, dirs, _ in os.walk(root_dir):
            for dir in dirs:
                if dir.startswith('SNGS'):
                    sngs_path = os.path.join(root, dir)
                    for filename in os.listdir(sngs_path):
                        file_path = os.path.join(sngs_path, filename)
                        if file_path.endswith('.json'):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                filtered_annotations = [
                                    ann for ann in data.get('annotations', [])
                                    if ann.get("attributes", {}).get("role") == "player"
                                    and ann["attributes"].get("jersey") is not None
                                ]
                                for ann in filtered_annotations:
                                    image_id = ann['image_id']
                                    if image_id in image_id_to_file:
                                        file_name = image_id_to_file[image_id]

                                        image_path = os.path.join(sngs_path, "img1", file_name)
                                        if image_path not in image_path_set:
                                            self.image_paths.append(image_path)
                                            image_path_set.add(image_path)
                                        self.annotations.append(ann)
                                    else:
                                        log.warning(f"Image ID {image_id} not found.")

                                    if (height in image_id_to_height) and (width in image_id_to_width):
                                        height = image_id_to_height[image_id] 
                                        width = image_id_to_width[image_id] 

                                        metainfo = {
                                            'height': height,
                                            'width': width,
                                        }
                                        self.metainfos.append(metainfo)
                                    else:
                                        log.warning(f"Image shape not found.")

        log.info(f"Loaded {len(self.image_paths)} unique images with annotations.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return data in the expected structured format."""
        try:
            image_path = self.image_paths[idx]
            annotation = self.annotations[idx]
            height, width = self.metainfos[idx]

            img_shape = (height, width, 3)
            # Load and transform the image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)

            # Create a structured data element for the sample
            data_sample = TextDetDataSample()
            gt_instances = InstanceData(img_shape=img_shape)

            bbox = annotation['bbox_image']
            x = bbox['x']
            y = bbox['y']
            x_center = bbox['x_center']
            y_center = bbox['y_center']
            w = bbox['w']
            h = bbox['h']

            gt_instances.bbox = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
            num_instances = len(gt_instances.bbox)  # Ensure bbox_image is a list or similar structure
            gt_instances.ignored = [False] * num_instances
            
            # gt_instances.polygons = torch.tensor([x_center, y_center, w, h, 0.0, 1.0]) # 4-point polygon for AABB
            gt_instances.polygons = torch.tensor([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=torch.float32)

            
            gt_instances.set_metainfo({
                'image_id': annotation['image_id'],
                'category_id': annotation['category_id'],
            })
            data_sample.gt_instances = gt_instances
            # Return the properly structured data
            return {'inputs': image_tensor, 'data_samples': data_sample}
        except IndexError as e:
            log.error(f"IndexError: {e}. Returning None for idx {idx}")
            return None