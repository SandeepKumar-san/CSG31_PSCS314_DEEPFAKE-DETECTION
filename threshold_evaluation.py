import torch
import numpy as np
from scipy import stats
import json
from model import DeepFakeDetector
from cached_dataloader import get_cached_dataloaders
from cpu_optimized_config import get_cpu_model_config

def evaluate_thresholds():
    """Evaluate model performance across multiple thresholds on full test set"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Load model
    model_config = get_cpu_model_config()
    model = DeepFakeDetector(
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load('best_deepfake_detector.pth', map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
    
    # Load test data
    _, _, test_loader = get_cached_dataloaders()
    if test_loader is None:
        print("❌ No test data found")
        return
    
    # Collect all predictions and labels
    all_predictions = []
    all_labels = []
    
    print("📊 Evaluating on full test set...")
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            probabilities = torch.sigmoid(outputs)
            
            all_predictions.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    print(f"📈 Total test samples: {len(all_predictions)}")
    print(f"   Real videos: {np.sum(all_labels == 0)}")
    print(f"   Fake videos: {np.sum(all_labels == 1)}")
    
    # Test thresholds
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    results = {}
    
    print("\n🎯 THRESHOLD EVALUATION RESULTS:")
    print("=" * 80)
    
    for threshold in thresholds:
        # Binary predictions at this threshold
        binary_preds = (all_predictions >= threshold).astype(int)
        
        # Separate real and fake samples
        fake_mask = (all_labels == 1)
        real_mask = (all_labels == 0)
        
        fake_predictions = all_predictions[fake_mask]
        real_predictions = all_predictions[real_mask]
        
        # Detection rates
        fake_detection_rate = np.mean(fake_predictions >= threshold)
        real_detection_rate = np.mean(real_predictions >= threshold)  # False positive rate
        
        # Discrimination power
        discrimination = fake_detection_rate - real_detection_rate
        
        # Statistical significance (t-test)
        fake_binary = (fake_predictions >= threshold).astype(int)
        real_binary = (real_predictions >= threshold).astype(int)
        t_stat, p_value = stats.ttest_ind(fake_binary, real_binary)
        
        # Overall accuracy
        accuracy = np.mean(binary_preds == all_labels)
        
        results[threshold] = {
            'fake_detection_rate': fake_detection_rate,
            'real_detection_rate': real_detection_rate,
            'discrimination_power': discrimination,
            'p_value': p_value,
            'accuracy': accuracy,
            'fake_samples': int(np.sum(fake_mask)),
            'real_samples': int(np.sum(real_mask))
        }
        
        print(f"Threshold {threshold:.1f}:")
        print(f"  Fake Detection Rate: {fake_detection_rate:.3f} ({fake_detection_rate*100:.1f}%)")
        print(f"  Real Detection Rate: {real_detection_rate:.3f} ({real_detection_rate*100:.1f}%)")
        print(f"  Discrimination Power: {discrimination:.3f}")
        print(f"  Statistical Significance: p={p_value:.6f}")
        print(f"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Samples: {int(np.sum(fake_mask))} fake, {int(np.sum(real_mask))} real")
        print()
    
    # Find optimal threshold
    best_threshold = max(results.keys(), key=lambda t: results[t]['discrimination_power'])
    best_result = results[best_threshold]
    
    print("🏆 OPTIMAL THRESHOLD SELECTION:")
    print("=" * 80)
    print(f"Best Threshold: {best_threshold}")
    print(f"Discrimination Power: {best_result['discrimination_power']:.3f}")
    print(f"Accuracy: {best_result['accuracy']:.3f} ({best_result['accuracy']*100:.1f}%)")
    print(f"Statistical Significance: p={best_result['p_value']:.6f}")
    
    # Save results
    output_data = {
        'evaluation_summary': {
            'total_samples': len(all_predictions),
            'fake_samples': int(np.sum(all_labels == 1)),
            'real_samples': int(np.sum(all_labels == 0)),
            'optimal_threshold': best_threshold,
            'optimal_accuracy': best_result['accuracy'],
            'optimal_discrimination': best_result['discrimination_power']
        },
        'threshold_results': results
    }
    
    with open('threshold_evaluation_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to threshold_evaluation_results.json")
    
    return results, best_threshold

if __name__ == "__main__":
    evaluate_thresholds()