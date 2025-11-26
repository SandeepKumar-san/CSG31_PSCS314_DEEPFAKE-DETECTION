# PROJECT CONFIGURATION FOR DEEPFAKE DETECTION
# Compatible with both CPU and GPU training

import torch
import multiprocessing as mp

# MODEL AND TRAINING CONFIGURATION
PROJECT_CONFIG = {
    # Data Loading Configuration
    'batch_size': 8,              # Balanced for CPU/GPU compatibility
    'num_workers': 2,             # Multiprocessing workers
    'pin_memory': True,           # Memory optimization
    
    # Model Architecture
    'sequence_length': 5,         # Video frames per sequence
    'hidden_size': 256,           # GRU hidden units
    'num_layers': 1,              # GRU layers
    'dropout': 0.2,               # Dropout rate
    
    # Training Hyperparameters
    'epochs': 25,                 # Training epochs
    'learning_rate': 2e-4,        # Learning rate
    'weight_decay': 5e-4,         # L2 regularization
    
    # Dataset Sampling (CRITICAL for 3300 videos)
    'sample_ratio': 0.2,          # Use 20% of manipulated data
    'max_videos_per_class': 600,  # Limit videos per class (300 original + 600 manipulated)
    'stratified_sampling': True,   # Balanced sampling
    
    # Processing Optimizations
    'face_detection_batch': 1,    # Process one frame at a time
    'feature_cache': True,        # Cache extracted features
    'low_memory_mode': True,      # Enable memory optimizations
    
    # Caching Configuration
    'use_cache': True,            # Use cached preprocessed faces
    'cache_dir': 's:/Capstone/Capstone/cached_faces',
    'preprocess_batch_size': 1,   # Batch size for preprocessing
    
    # System Threading
    'torch_threads': mp.cpu_count() // 2,  # CPU thread count
    'opencv_threads': 2,          # OpenCV threads
}

def setup_environment():
    """Configure PyTorch environment for CPU/GPU compatibility"""
    # Set CPU thread count
    torch.set_num_threads(PROJECT_CONFIG['torch_threads'])
    
    # Enable CPU optimizations (also benefits GPU)
    torch.backends.mkldnn.enabled = True
    
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"✅ Environment configured for {device}:")
    print(f"   - Threads: {PROJECT_CONFIG['torch_threads']}")
    print(f"   - Batch size: {PROJECT_CONFIG['batch_size']}")
    print(f"   - Sample ratio: {PROJECT_CONFIG['sample_ratio']}")
    print(f"   - Max videos per class: {PROJECT_CONFIG['max_videos_per_class']}")

def get_model_config():
    """Return model architecture configuration"""
    return {
        'sequence_length': PROJECT_CONFIG['sequence_length'],
        'hidden_size': PROJECT_CONFIG['hidden_size'], 
        'num_layers': PROJECT_CONFIG['num_layers'],
        'dropout': PROJECT_CONFIG['dropout']
    }

def get_training_config():
    """Return training configuration"""
    return {
        'batch_size': PROJECT_CONFIG['batch_size'],
        'epochs': PROJECT_CONFIG['epochs'],
        'learning_rate': PROJECT_CONFIG['learning_rate'],
        'weight_decay': PROJECT_CONFIG['weight_decay'],
        'num_workers': PROJECT_CONFIG['num_workers'],
        'pin_memory': PROJECT_CONFIG['pin_memory']
    }

# Backward compatibility aliases
CPU_CONFIG = PROJECT_CONFIG
setup_cpu_environment = setup_environment
get_cpu_model_config = get_model_config
get_cpu_training_config = get_training_config