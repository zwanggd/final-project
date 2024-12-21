import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class TrackletDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def load_finetuned_models(model_paths, device='cuda'):
    models = {}
    model_depths = {
        'svtr-tiny': 18,
        'svtr-small': 34,
        'nrtr': 50,
        'satrn': 101,
        'aster': 152
    }
    
    for model_name, model_path in model_paths.items():
        depth = model_depths[model_name]
        model_arch = getattr(models, f'resnet{depth}')()
        model_arch.fc = nn.Linear(model_arch.fc.in_features, 100)
        model_arch.load_state_dict(torch.load(model_path, map_location=device))
        model_arch = model_arch.to(device)
        model_arch.eval()
        models[model_name] = model_arch
    return models

def predict_with_majority_voting(models, dataloader, device='cuda'):
    predictions_per_model = {model_name: [] for model_name in models}
    
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            for model_name, model in models.items():
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predictions_per_model[model_name].extend(predicted.cpu().numpy())
    
    final_predictions = []
    for i in range(len(dataloader.dataset)):
        frame_predictions = [predictions_per_model[model_name][i] for model_name in models]
        majority_vote = np.bincount(frame_predictions).argmax()
        final_predictions.append(majority_vote)
    
    final_tracklet_prediction = np.bincount(final_predictions).argmax()
    return final_tracklet_prediction

if __name__ == "__main__":
    model_paths = {
        'svtr-tiny': '/Users/kai/GSR/soccernet/mmocr/weights/svtr-tiny_final.pth',
        'svtr-small': '/Users/kai/GSR/soccernet/mmocr/weights/svtr-small_final.pth',
        'nrtr': '/Users/kai/GSR/soccernet/mmocr/weights/nrtr_final.pth',
        'satrn': '/Users/kai/GSR/soccernet/mmocr/weights/satrn_final.pth',
        'aster': '/Users/kai/GSR/soccernet/mmocr/weights/aster_final.pth'
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = load_finetuned_models(model_paths, device)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    tracklet_dataset = TrackletDataset(transform=transform)
    tracklet_dataloader = DataLoader(tracklet_dataset, batch_size=4, shuffle=False)
    
    final_prediction = predict_with_majority_voting(models, tracklet_dataloader, device)
    print(final_prediction)
