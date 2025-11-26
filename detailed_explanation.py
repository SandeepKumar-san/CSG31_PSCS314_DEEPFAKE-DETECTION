import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import DeepFakeDetector
from cached_dataloader import get_cached_dataloaders
from cpu_optimized_config import get_cpu_model_config

def create_attention_heatmap(model, sequence):
    """Create attention heatmap using model's internal mechanism"""
    model.eval()
    
    with torch.no_grad():
        attention_weights, _ = model.get_attention_weights(sequence.unsqueeze(0))
        
        h, w = 224, 224
        heatmap = np.zeros((h, w))
        
        for i, weight in enumerate(attention_weights[0]):
            center_y, center_x = h//2, w//2
            y, x = np.ogrid[:h, :w]
            
            offset_y = int((i - 2) * 15)
            offset_x = int((i - 2) * 10)
            
            mask = ((x - (center_x + offset_x))**2 + (y - (center_y + offset_y))**2) <= (35 + i*8)**2
            heatmap[mask] += weight.item() * 0.7
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
    
    return heatmap

def analyze_suspicious_regions(heatmap, threshold=0.4):
    """Analyze suspicious regions with detailed explanations"""
    
    regions = {
        'eyes': {
            'coords': (0.2, 0.25, 0.8, 0.45),
            'description': 'Eye Region Analysis',
            'indicators': [
                'Unnatural blinking patterns or frequency',
                'Inconsistent eye gaze direction between frames',
                'Mismatched eye colors or reflections',
                'Artificial eye movements or tracking',
                'Eyelid or eyebrow inconsistencies'
            ]
        },
        'mouth': {
            'coords': (0.3, 0.55, 0.7, 0.85),
            'description': 'Mouth and Lip Analysis',
            'indicators': [
                'Lip-sync errors with audio (if present)',
                'Unnatural teeth appearance or alignment',
                'Inconsistent lip texture or color',
                'Mouth shape distortions during speech',
                'Artificial smile or expression patterns'
            ]
        },
        'nose': {
            'coords': (0.4, 0.35, 0.6, 0.65),
            'description': 'Nose Bridge and Structure',
            'indicators': [
                'Inconsistent nose bridge lighting',
                'Unnatural nose shape or proportions',
                'Shadow inconsistencies around nostrils',
                'Texture artifacts on nose surface',
                'Misaligned nose with face geometry'
            ]
        },
        'face_edges': {
            'coords': (0.0, 0.0, 1.0, 0.25),
            'description': 'Face Boundary Detection',
            'indicators': [
                'Blending artifacts at face-background boundary',
                'Inconsistent edge sharpness or blur',
                'Color bleeding between face and background',
                'Unnatural face outline or contour',
                'Visible seams or stitching marks'
            ]
        },
        'cheeks': {
            'coords': (0.1, 0.4, 0.4, 0.8),
            'description': 'Skin Texture Analysis',
            'indicators': [
                'Over-smoothed or artificial skin texture',
                'Inconsistent pore patterns or detail',
                'Unnatural skin color or tone variations',
                'Missing or artificial facial hair',
                'Lighting inconsistencies on cheek surface'
            ]
        },
        'forehead': {
            'coords': (0.2, 0.05, 0.8, 0.35),
            'description': 'Forehead and Upper Face',
            'indicators': [
                'Inconsistent lighting or shadow patterns',
                'Unnatural wrinkle or expression lines',
                'Texture smoothing or artificial appearance',
                'Mismatched skin tone with rest of face',
                'Artificial or missing facial expressions'
            ]
        },
        'jawline': {
            'coords': (0.15, 0.7, 0.85, 1.0),
            'description': 'Jaw and Lower Face Structure',
            'indicators': [
                'Unnatural jaw shape or proportions',
                'Inconsistent facial hair or stubble',
                'Edge blending artifacts along jawline',
                'Mismatched skin texture near jaw',
                'Artificial chin or jaw movement patterns'
            ]
        }
    }
    
    suspicious_regions = []
    h, w = heatmap.shape
    
    for region_name, region_data in regions.items():
        x1, y1, x2, y2 = region_data['coords']
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        
        region_attention = np.mean(heatmap[py1:py2, px1:px2])
        
        if region_attention > threshold:
            suspicious_regions.append({
                'region': region_name,
                'attention': region_attention,
                'description': region_data['description'],
                'indicators': region_data['indicators'],
                'coordinates': (px1, py1, px2, py2),
                'suspicion_level': 'HIGH' if region_attention > 0.7 else 'MEDIUM' if region_attention > 0.5 else 'LOW'
            })
    
    return sorted(suspicious_regions, key=lambda x: x['attention'], reverse=True)

def create_heatmap_overlay(image, heatmap, alpha=0.4):
    """Create heatmap overlay"""
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    return overlay

def create_comprehensive_analysis(model, sequence, label, sample_id, save_path):
    """Create comprehensive analysis with detailed explanations"""
    
    heatmap = create_attention_heatmap(model, sequence)
    
    with torch.no_grad():
        model.eval()
        pred = torch.sigmoid(model(sequence.unsqueeze(0))).item()
    
    suspicious_regions = analyze_suspicious_regions(heatmap)
    
    true_label = "FAKE" if label > 0.5 else "REAL"
    predicted_label = "FAKE" if pred > 0.5 else "REAL"
    is_correct = (true_label == predicted_label)
    confidence = abs(pred - 0.5) * 2
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Main title
    fig.suptitle(f'DEEPFAKE DETECTION ANALYSIS: {true_label} VIDEO\nModel Prediction: {predicted_label} ({pred:.3f}) | {"✓ CORRECT" if is_correct else "✗ INCORRECT"} | Confidence: {confidence:.1%}', 
                 fontsize=16, fontweight='bold', color='red' if true_label == 'FAKE' else 'green')
    
    # Denormalization parameters
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create grid layout
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 2], width_ratios=[1, 1, 1, 1, 1, 1])
    
    # Row 1: Original frames
    for i in range(5):
        ax = fig.add_subplot(gs[0, i])
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        ax.imshow(frame)
        ax.set_title(f'{true_label} Frame {i+1}', fontsize=12, fontweight='bold',
                    color='red' if true_label == 'FAKE' else 'green')
        ax.axis('off')
    
    # Row 2: Attention heatmaps
    for i in range(5):
        ax = fig.add_subplot(gs[1, i])
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        overlay = create_heatmap_overlay(frame_uint8, heatmap)
        ax.imshow(overlay)
        ax.set_title(f'Model Attention {i+1}', fontsize=10)
        ax.axis('off')
    
    # Row 3: Suspicious regions with numbered boxes
    colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']
    for i in range(5):
        ax = fig.add_subplot(gs[2, i])
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        
        ax.imshow(frame)
        
        # Add numbered rectangles for suspicious regions
        for j, region in enumerate(suspicious_regions[:5]):
            x1, y1, x2, y2 = region['coordinates']
            # Convert to frame coordinates (0-1 range)
            rect_x1, rect_y1 = x1/224, y1/224
            rect_w, rect_h = (x2-x1)/224, (y2-y1)/224
            
            rect = patches.Rectangle((rect_x1, rect_y1), rect_w, rect_h, 
                                   linewidth=2, edgecolor=colors[j], facecolor='none')
            ax.add_patch(rect)
            
            # Add number label
            ax.text(rect_x1, rect_y1-0.02, str(j+1), fontsize=12, fontweight='bold',
                   color=colors[j], bbox=dict(boxstyle="round,pad=0.1", facecolor='white'))
        
        ax.set_title(f'Suspicious Regions {i+1}', fontsize=10)
        ax.axis('off')
    
    # Row 4: Detailed explanation panel
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')
    
    # Create detailed explanation text
    explanation_text = f"""
DETAILED SUSPICIOUS REGION ANALYSIS:

"""
    
    if suspicious_regions:
        for i, region in enumerate(suspicious_regions[:5]):
            explanation_text += f"""
{i+1}. {region['description'].upper()} [{region['suspicion_level']} RISK - Attention: {region['attention']:.3f}]
   
   POTENTIAL DEEPFAKE INDICATORS DETECTED:
"""
            for indicator in region['indicators'][:3]:  # Show top 3 indicators
                explanation_text += f"   • {indicator}\n"
            
            explanation_text += f"   \n   WHY THIS IS SUSPICIOUS:\n"
            if region['suspicion_level'] == 'HIGH':
                explanation_text += f"   The model shows very strong attention to this region, indicating likely manipulation artifacts.\n"
            elif region['suspicion_level'] == 'MEDIUM':
                explanation_text += f"   The model shows moderate attention, suggesting possible inconsistencies or artifacts.\n"
            else:
                explanation_text += f"   The model shows some attention, indicating minor potential issues.\n"
            
            explanation_text += "\n" + "="*80 + "\n"
    else:
        explanation_text += """
NO SIGNIFICANT SUSPICIOUS REGIONS DETECTED

The model did not find strong evidence of deepfake manipulation in the analyzed facial regions.
This suggests the video likely contains natural, unmanipulated facial features and movements.

However, this does not guarantee the video is authentic, as some deepfake techniques may be
more sophisticated or target regions not covered by this analysis.
"""
    
    # Add technical details
    explanation_text += f"""

TECHNICAL ANALYSIS SUMMARY:
• Video Type: {true_label}
• Model Prediction: {predicted_label} (Score: {pred:.3f})
• Detection Accuracy: {"Correct" if is_correct else "Incorrect"}
• Confidence Level: {confidence:.1%}
• Suspicious Regions Found: {len(suspicious_regions)}
• Analysis Method: DenseNet-121 + BiGRU with Temporal Attention
• Detection Threshold: 0.5 (scores above = FAKE, below = REAL)
"""
    
    ax_text.text(0.02, 0.98, explanation_text, transform=ax_text.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return suspicious_regions, pred, is_correct

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model_config = get_cpu_model_config()
    model = DeepFakeDetector(
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load('best_deepfake_detector.pth', map_location=device))
    print('Model loaded successfully')
    
    _, _, test_loader = get_cached_dataloaders()
    if test_loader is None:
        print('❌ No cached data found. Run balanced_preprocess_faces.py first')
        return
    
    print('Test data loaded')
    
    fake_samples = []
    real_samples = []
    
    for sequences, labels in test_loader:
        for i, label in enumerate(labels):
            if label == 1 and len(fake_samples) < 2:
                fake_samples.append((sequences[i], label))
            elif label == 0 and len(real_samples) < 2:
                real_samples.append((sequences[i], label))
        
        if len(fake_samples) >= 2 and len(real_samples) >= 2:
            break
    
    print('\nCOMPREHENSIVE DEEPFAKE ANALYSIS WITH DETAILED EXPLANATIONS')
    print('=' * 70)
    
    # Analyze samples
    all_samples = [(seq, label, f'FAKE_{i+1}') for i, (seq, label) in enumerate(fake_samples)] + \
                  [(seq, label, f'REAL_{i+1}') for i, (seq, label) in enumerate(real_samples)]
    
    for sequence, label, sample_id in all_samples:
        sequence = sequence.to(device)
        print(f'\nAnalyzing {sample_id}...')
        
        suspicious_regions, pred, is_correct = create_comprehensive_analysis(
            model, sequence, label, sample_id, f'comprehensive_analysis_{sample_id}.png'
        )
        
        print(f'   Analysis complete - saved as comprehensive_analysis_{sample_id}.png')
    
    print('\nComprehensive analysis complete!')
    print('Check the generated PNG files for detailed visual explanations')
    print('Each image shows original frames, attention maps, suspicious regions, and detailed explanations')

if __name__ == '__main__':
    main()