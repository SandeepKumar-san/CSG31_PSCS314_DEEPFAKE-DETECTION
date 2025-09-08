import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import DeepFakeDetector
from dataloader import create_dataloaders

def evaluate_model(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    original_dir = './data/DFD_original sequences'
    manipulated_dir = './data/DFD_manipulated_sequences'
    model_path = 'best_model.pth'
    batch_size = 2
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print('GPU available but will use automatic detection')
        torch.cuda.empty_cache()
    else:
        print('Using CPU for evaluation')
    
    # Load model
    print('Loading trained model...')
    model = DeepFakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Create test dataloader
    print('Creating test dataloader...')
    _, _, test_loader = create_dataloaders(
        original_dir, manipulated_dir, batch_size=batch_size
    )
    
    # Evaluate
    print('Evaluating model on test set...')
    predictions, probabilities, labels = evaluate_model(model, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Check class distribution
    unique_labels = np.unique(labels)
    print(f'\nClass distribution in test set:')
    print(f'REAL (0): {np.sum(labels == 0)} samples')
    print(f'FAKE (1): {np.sum(labels == 1)} samples')
    
    # Calculate ROC-AUC with error handling
    try:
        roc_auc = roc_auc_score(labels, probabilities)
        roc_auc_available = True
    except ValueError as e:
        print(f'\nROC-AUC Error: {e}')
        roc_auc = None
        roc_auc_available = False
    
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    if roc_auc_available:
        print(f'ROC-AUC Score: {roc_auc:.4f}')
    else:
        print('ROC-AUC Score: Not available (only one class in test set)')
    print('\nClassification Report:')
    print(classification_report(labels, predictions, target_names=['REAL', 'FAKE']))
    
    # Plot confusion matrix
    plot_confusion_matrix(labels, predictions)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc) if roc_auc_available else None,
        'total_samples': len(labels),
        'correct_predictions': int(np.sum(predictions == labels)),
        'real_samples': int(np.sum(labels == 0)),
        'fake_samples': int(np.sum(labels == 1))
    }
    
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\nResults saved to evaluation_results.json')

if __name__ == '__main__':
    main()