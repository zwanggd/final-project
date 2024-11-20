from mmocr.datasets.nested_ocr_dataset import NestedOCRDataset
import logging
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to display info logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Output logs to console
)
log = logging.getLogger(__name__)  # Use __name__ to identify the logger

train_dataset = NestedOCRDataset(root_dir='/Users/kai/GSR/data/SoccerNetGS/train')
val_dataset = NestedOCRDataset(root_dir='/Users/kai/GSR/data/SoccerNetGS/valid')

# Define the DataLoader with the custom collate function
def custom_collate_fn(batch):
    """Ensure batch data is structured correctly for the model's preprocessor."""
    batch = [item for item in batch if item is not None]  # Filter out None

    # Split inputs and annotations
    inputs = [item['inputs'] for item in batch]
    annotations = [item['annotations'] for item in batch]

    return {
        'inputs': inputs,         # List of input paths or tensors
        'data_samples': annotations  # List of corresponding annotations
    }

# Create DataLoaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=2,  # Use a small batch size for testing
    shuffle=True, 
    collate_fn=custom_collate_fn
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=2, 
    shuffle=False, 
    collate_fn=custom_collate_fn
)

# Test the first batch from the train loader
for i, batch in enumerate(train_loader):
    log.info(f"Batch {i}: {batch}")  # Print the structure of the batch
    if i >= 1:  # Stop after 2 batches for testing
        break

# Check if the batch contains 'inputs' and 'data_samples'
assert 'inputs' in batch, "Batch is missing 'inputs'."
assert 'data_samples' in batch, "Batch is missing 'data_samples'."
log.info("Batch structure is correct.")

