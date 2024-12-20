import os
from mmocr.apis import train_text_recognizer
from mmocr.utils import register_all_modules

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def validate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_loss /= len(val_dataloader)
    val_accuracy = correct / total * 100
    
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    return val_loss, val_accuracy

def test_model(model, test_dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_loss /= len(test_dataloader)
    test_accuracy = correct / total * 100
    
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy

def train_model(model, train_dataloader, val_dataloader, test_dataloader, epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}')
        
        val_loss, val_accuracy = validate_model(model, val_dataloader, criterion, device)
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
        
    test_loss, test_accuracy = test_model(model, test_dataloader, criterion, device)
    print(f'Final Test Accuracy: {test_accuracy:.2f}%')

def create_finetuning_config(model_name, train_dataset_path, valid_images_path, ground_truth_path, pretrained_weight_path, epochs=20, batch_size=32):
    config = {
        'model': {
            'type': model_name,
            'backbone': {
                'type': 'ResNet',
                'depth': 18 if model_name == 'svtr_tiny' else 34 if model_name == 'svtr_small' else 50 if model_name == 'satrn' else 101 if model_name == 'nrtr' else 152
            },
            'neck': {
                'type': 'FPN',
                'in_channels': [64, 128, 256, 512],
                'out_channels': 256
            },
            'head': {
                'type': 'CTCHead',
                'in_channels': 256,
                'num_classes': 100  
            }
        },
        'load_from': pretrained_weight_path,
        'train_dataloader': {
            'batch_size': batch_size,
            'num_workers': 8,
            'shuffle': True,
            'dataset': {
                'type': 'OCRDataset',
                'data_root': train_dataset_path,
                'ann_file': ground_truth_path,
                'img_prefix': train_dataset_path
            }
        },
        'val_dataloader': {
            'batch_size': batch_size,
            'num_workers': 4,
            'shuffle': False,
            'dataset': {
                'type': 'OCRDataset',
                'data_root': train_dataset_path,
                'ann_file': ground_truth_path,
                'img_prefix': train_dataset_path
            }
        },
        'val_evaluator': {
            'type': 'AccEvaluator'
        },
        'optimizer': {
            'type': 'Adam',
            'lr': 0.0001,
            'weight_decay': 0.05
        },
        'param_scheduler': [
            {'type': 'CosineAnnealingLR', 'T_max': epochs}
        ],
        'train_cfg': {
            'type': 'EpochBasedTrainLoop',
            'max_epochs': epochs
        },
        'log_config': {
            'interval': 50,
            'hooks': [
                {'type': 'TextLoggerHook'}
            ]
        },
        'default_hooks': {
            'checkpoint': {
                'type': 'CheckpointHook',
                'interval': 1
            }
        }
    }
    return config


def main():
    train_dataset_path = '/Users/kai/GSR/data/Train'
    valid_images_path = '/Users/kai/GSR/data/Valid'
    ground_truth_path = '/Users/kai/GSR/data/Train/gt.json'
    work_dir = 'work_dirs'
    model_names = ['svtr_small', 'svtr_tiny', 'satrn', 'nrtr', 'aster']
    
    # Register MMOCR modules before training
    register_all_modules()
    
    for model_name in model_names:
        # Create configuration for each model
        config = create_finetuning_config(
            model_name=model_name,
            train_dataset_path=train_dataset_path,
            valid_images_path=valid_images_path,
            ground_truth_path=ground_truth_path,
            work_dir=os.path.join(work_dir, model_name),
            epochs=20,
            batch_size=32
        )
        
        # Actual training using MMOCR's train_text_recognizer
        train_text_recognizer(config)

if __name__ == "__main__":
    main()