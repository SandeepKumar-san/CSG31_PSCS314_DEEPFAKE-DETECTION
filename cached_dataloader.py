import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from cpu_optimized_config import get_cpu_training_config

class CachedFaceDataset(Dataset):
    """Dataset that loads preprocessed cached faces"""
    
    def __init__(self, cached_data):
        self.data = cached_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        faces, label = self.data[idx]
        return faces, torch.tensor(label, dtype=torch.long)

def load_cached_data():
    """Load preprocessed faces from cache"""
    cache_file = "s:/Capstone/Capstone/cached_faces/balanced_preprocessed_faces.pkl"
    
    if not os.path.exists(cache_file):
        print("Cache file not found. Run preprocess_faces.py first.")
        return None, None, None
    
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    
    print(f"Loaded {len(cached_data)} cached videos")
    
    # Split data
    train_data, temp_data = train_test_split(cached_data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    return train_data, val_data, test_data

def get_cached_dataloaders():
    """Get dataloaders for cached preprocessed faces"""
    train_data, val_data, test_data = load_cached_data()
    
    if train_data is None:
        return None, None, None
    
    config = get_cpu_training_config()
    
    # Create datasets
    train_dataset = CachedFaceDataset(train_data)
    val_dataset = CachedFaceDataset(val_data)
    test_dataset = CachedFaceDataset(test_data)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    
    print(f"Created dataloaders:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples") 
    print(f"   Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader