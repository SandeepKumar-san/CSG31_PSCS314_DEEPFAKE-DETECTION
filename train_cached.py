import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from model import DeepFakeDetector
from cached_dataloader import get_cached_dataloaders
from cpu_optimized_config import setup_cpu_environment, get_cpu_model_config, get_cpu_training_config

def train_with_cache():
    """Train model using cached preprocessed faces"""
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    setup_cpu_environment()
    
    # Load cached data
    train_loader, val_loader, test_loader = get_cached_dataloaders()
    if train_loader is None:
        print("❌ Run preprocess_faces.py first to create cache")
        return
    
    # Initialize model
    model_config = get_cpu_model_config()
    train_config = get_cpu_training_config()
    
    model = DeepFakeDetector(
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    
    # Add learning rate scheduler and early stopping
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Early stopping parameters
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
    # Training history tracking
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"🚀 Starting cached training...")
    print(f"   Epochs: 25 (with early stopping)")
    print(f"   Early stopping patience: 5 epochs")
    print(f"   Batch size: {train_config['batch_size']}")
    
    # Training loop
    for epoch in range(25):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for faces, labels in pbar:
            faces, labels = faces.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Convert labels to float for BCEWithLogitsLoss
            labels = labels.float()
            
            outputs = model(faces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for faces, labels in val_loader:
                faces, labels = faces.to(device), labels.to(device)
                labels = labels.float()
                outputs = model(faces)
                val_loss += criterion(outputs, labels).item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Save to history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc / 100)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc / 100)
        
        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_deepfake_detector.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")
            
        # Early stopping check
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Save model and training history
    torch.save(model.state_dict(), 'deepfake_detector_cached.pth')
    
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("✅ Model saved as deepfake_detector_cached.pth")
    print("📊 Training history saved as training_history.json")

if __name__ == "__main__":
    train_with_cache()