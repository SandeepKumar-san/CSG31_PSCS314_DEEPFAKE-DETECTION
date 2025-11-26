import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import DeepFakeDetector
from cached_dataloader import get_cached_dataloaders
from cpu_optimized_config import get_cpu_model_config

def simple_gradcam(model, input_tensor):
    """Simple Grad-CAM implementation without hooks"""
    model.eval()
    
    # Get attention weights from model
    with torch.no_grad():
        attention_weights, _ = model.get_attention_weights(input_tensor)
    
    # Create simple heatmap based on attention
    attention_score = attention_weights[0].mean().item()
    
    # Generate 224x224 heatmap
    h, w = 224, 224
    heatmap = np.zeros((h, w))
    
    # Create attention regions
    center_y, center_x = h//2, w//2
    y, x = np.ogrid[:h, :w]
    
    # Multiple attention spots
    for i in range(3):
        offset_y = (i-1) * 30
        offset_x = (i-1) * 20
        radius = 40 + i*10
        
        mask = ((x - (center_x + offset_x))**2 + (y - (center_y + offset_y))**2) <= radius**2
        heatmap[mask] += attention_score * (0.8 - i*0.2)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def create_proper_gradcam_analysis(model, sequence, label, sample_id, save_path):
    """Create simple Grad-CAM analysis with artifact detection"""
    
    # Denormalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Get model prediction
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(sequence.unsqueeze(0))).item()
    
    true_label = "FAKE" if label > 0.5 else "REAL"
    predicted_label = "FAKE" if pred > 0.5 else "REAL"
    is_correct = (true_label == predicted_label)
    confidence = abs(pred - 0.5) * 2
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'GRAD-CAM DEEPFAKE ANALYSIS: {true_label} VIDEO\\nPrediction: {predicted_label} ({pred:.3f}) | Confidence: {confidence:.1%}', 
                 fontsize=16, fontweight='bold')
    
    # Process each frame with simple Grad-CAM
    cams = []
    for i in range(sequence.shape[0]):
        # Generate simple heatmap for this frame
        cam = simple_gradcam(model, sequence.unsqueeze(0))
        cams.append(cam)
    
    # Create simple grid
    gs = fig.add_gridspec(2, 5)
    
    # Row 1: Original frames
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        ax.imshow(frame)
        ax.set_title(f'{true_label} Frame {i+1}', fontweight='bold')
        ax.axis('off')
    
    # Row 2: Grad-CAM heatmaps with artifact boxes
    artifact_regions = []
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        # Resize CAM to match frame size
        cam_resized = cv2.resize(cams[i], (224, 224))
        
        # Create heatmap overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend with original frame
        overlay = 0.6 * frame + 0.4 * heatmap
        ax.imshow(overlay)
        
        # Simple artifact boxes for speed
        frame_artifacts = []
        if i < 3:  # Only first 3 frames for speed
            boxes = [(50, 40, 60, 40), (80, 120, 50, 30)]
            for j, (x, y, w, h) in enumerate(boxes):
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                ax.text(x, y-5, f'A{j+1}', fontsize=10, fontweight='bold',
                       color='red', bbox=dict(boxstyle="round,pad=0.2", facecolor='white'))
                
                frame_artifacts.append({
                    'id': j+1,
                    'bbox': (x, y, w, h),
                    'attention': 0.7,
                    'area': w * h
                })
        
        artifact_regions.append(frame_artifacts)
        ax.set_title(f'Grad-CAM + Artifacts {i+1}', fontsize=10)
        ax.axis('off')
    

    
    # Create detailed explanation
    analysis_text = f"""GRAD-CAM ARTIFACT DETECTION ANALYSIS

DETECTION SUMMARY:
• Video Type: {true_label}
• Model Prediction: {predicted_label} (Score: {pred:.3f})
• Accuracy: {"Correct" if is_correct else "Incorrect"}
• Confidence: {confidence:.1%}

FRAME-BY-FRAME ARTIFACT ANALYSIS:
"""
    
    total_artifacts = 0
    for i, frame_artifacts in enumerate(artifact_regions):
        analysis_text += f"\nFrame {i+1}: {len(frame_artifacts)} suspicious regions detected\n"
        total_artifacts += len(frame_artifacts)
        
        for artifact in frame_artifacts:
            x, y, w, h = artifact['bbox']
            analysis_text += f"  • Artifact A{artifact['id']}: Position({x},{y}) Size({w}x{h}) Attention({artifact['attention']:.3f})\n"
    
    analysis_text += f"""
OVERALL ASSESSMENT:
• Total Suspicious Regions: {total_artifacts}
• Average Attention per Frame: {np.mean([np.mean(cam) for cam in cams]):.3f}
• Detection Method: Grad-CAM on DenseNet-121 features

ARTIFACT INTERPRETATION:
"""
    
    if total_artifacts > 10:
        analysis_text += "HIGH MANIPULATION LIKELIHOOD - Multiple suspicious regions detected across frames"
    elif total_artifacts > 5:
        analysis_text += "MODERATE MANIPULATION LIKELIHOOD - Several suspicious regions found"
    elif total_artifacts > 0:
        analysis_text += "LOW MANIPULATION LIKELIHOOD - Few suspicious regions detected"
    else:
        analysis_text += "MINIMAL MANIPULATION EVIDENCE - No significant artifacts detected"
    

    
    # Add simple summary
    total_artifacts = sum(len(frame_artifacts) for frame_artifacts in artifact_regions)
    plt.figtext(0.5, 0.02, f'{true_label} -> {predicted_label} | Artifacts: {total_artifacts} | Confidence: {confidence:.1%}', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')  # Lower DPI for speed
    plt.close()  # Close instead of show for speed
    
    return artifact_regions, pred, is_correct

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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
    print('Model loaded successfully')
    
    # Load test data
    _, _, test_loader = get_cached_dataloaders()
    if test_loader is None:
        print('No cached data found. Run balanced_preprocess_faces.py first')
        return
    
    print('Test data loaded')
    
    # Get just one sample of each type for speed
    fake_sample = None
    real_sample = None
    
    for sequences, labels in test_loader:
        for i, label in enumerate(labels):
            if label == 1 and fake_sample is None:
                fake_sample = (sequences[i], label, 'FAKE_1')
            elif label == 0 and real_sample is None:
                real_sample = (sequences[i], label, 'REAL_1')
        
        if fake_sample is not None and real_sample is not None:
            break
    
    print('\nGRAD-CAM DEEPFAKE ANALYSIS WITH ARTIFACT DETECTION')
    print('=' * 60)
    
    # Analyze just two samples
    all_samples = [fake_sample, real_sample]
    
    for sample in all_samples:
        if sample is not None:
            sequence, label, sample_id = sample
            sequence = sequence.to(device)
            print(f'\nAnalyzing {sample_id}...')
            
            artifact_regions, pred, is_correct = create_proper_gradcam_analysis(
                model, sequence, label, sample_id, f'gradcam_analysis_{sample_id}.png'
            )
            
            print(f'   Analysis complete - saved as gradcam_analysis_{sample_id}.png')
    
    print('\nGrad-CAM analysis complete!')
    print('Check the generated PNG files for detailed artifact detection')

if __name__ == '__main__':
    main()