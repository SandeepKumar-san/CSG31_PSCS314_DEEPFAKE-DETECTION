import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json

from model import DeepFakeDetector
from dataloader import create_dataloaders

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    return total_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def main():
    # Configuration
    original_dir = './data/DFD_original sequences'
    manipulated_dir = './data/DFD_manipulated_sequences'
    batch_size = 2  # Restored batch size
    num_epochs = 50
    learning_rate = 1e-4
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU Name: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA not available, using CPU')
    
    # Create dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader, test_loader = create_dataloaders(
        original_dir, manipulated_dir, batch_size=batch_size
    )
    
    # Initialize model
    model = DeepFakeDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Clear cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save history
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.4f}')
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f'\nTraining completed. Best validation accuracy: {best_val_acc:.4f}')

if __name__ == '__main__':
    main()