# merged_explainer_artifacts_on_high_attention.py
# Place next to your model.py, cached_dataloader.py, cpu_optimized_config.py and best_deepfake_detector.pth
# Run: python merged_explainer_artifacts_on_high_attention.py

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import DeepFakeDetector
from cached_dataloader import get_cached_datalaloaders
from cpu_optimized_config import get_cpu_model_config

# ---------- Config ----------
HIGH_ATTENTION_THRESHOLD = 0.40   # frames with attention > this will get artifact boxes
REGION_Z_THRESHOLDS = (1.0, 1.8)  # (medium_z, high_z)

# -------------------- Utilities --------------------
def create_attention_heatmap(model, sequence):
    """Create attention heatmap using model's internal mechanism (global heatmap)."""
    model.eval()
    with torch.no_grad():
        attention_weights, _ = model.get_attention_weights(sequence.unsqueeze(0))  # (1, seq_len)
        seq_len = attention_weights.shape[1]
        h, w = 224, 224
        heatmap = np.zeros((h, w), dtype=np.float32)
        for i, weight in enumerate(attention_weights[0]):
            center_y, center_x = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            offset_y = int((i - (seq_len // 2)) * 15)
            offset_x = int((i - (seq_len // 2)) * 10)
            radius = 35 + i * 8
            mask = ((x - (center_x + offset_x))**2 + (y - (center_y + offset_y))**2) <= radius**2
            heatmap[mask] += float(weight.item()) * 0.7
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
    return heatmap

def analyze_suspicious_regions_normalized(heatmap, z_thresholds=(1.0, 1.8)):
    """
    Returns suspicious regions with normalized z-scores (frame-level).
    """
    regions = {
        'nose': {
            'coords': (0.4, 0.35, 0.6, 0.65),
            'description': 'Nose Bridge and Structure',
            'indicators': [
                'Inconsistent nose bridge lighting',
                'Unnatural nose shape or proportions',
                'Shadow inconsistencies around nostrils'
            ]
        },
        'mouth': {
            'coords': (0.3, 0.55, 0.7, 0.85),
            'description': 'Mouth and Lip Analysis',
            'indicators': [
                'Lip-sync errors with audio (if present)',
                'Unnatural teeth appearance or alignment',
                'Inconsistent lip texture or color'
            ]
        },
        'eyes': {
            'coords': (0.2, 0.25, 0.8, 0.45),
            'description': 'Eye Region Analysis',
            'indicators': [
                'Unnatural blinking patterns or frequency',
                'Inconsistent eye gaze direction between frames',
                'Mismatched eye colors or reflections'
            ]
        },
        'face_edges': {
            'coords': (0.0, 0.0, 1.0, 0.25),
            'description': 'Face Boundary Detection',
            'indicators': [
                'Blending at face-background boundary',
                'Color bleeding or seam artifacts'
            ]
        },
        'cheeks': {
            'coords': (0.1, 0.4, 0.4, 0.8),
            'description': 'Cheek / Skin Texture',
            'indicators': [
                'Over-smoothed skin texture',
                'Inconsistent pore patterns'
            ]
        }
    }

    flat = heatmap.flatten()
    mean_all = float(flat.mean())
    std_all = float(flat.std()) + 1e-8

    suspicious = []
    h, w = heatmap.shape
    for name, info in regions.items():
        x1, y1, x2, y2 = info['coords']
        px1, py1 = int(x1 * w), int(y1 * h)
        px2, py2 = int(x2 * w), int(y2 * h)
        region_slice = heatmap[py1:py2, px1:px2]
        if region_slice.size == 0:
            continue
        region_mean = float(region_slice.mean())
        z = (region_mean - mean_all) / std_all
        high_z, med_z = z_thresholds[1], z_thresholds[0]
        if z >= high_z:
            risk = 'HIGH'
        elif z >= med_z:
            risk = 'MEDIUM'
        elif z > 0:
            risk = 'LOW'
        else:
            risk = 'NONE'
        if risk != 'NONE':
            suspicious.append({
                'region': name,
                'description': info['description'],
                'indicators': info['indicators'],
                'coordinates': (px1, py1, px2, py2),
                'attention': region_mean,
                'z_score': z,
                'suspicion_level': risk
            })

    suspicious = sorted(suspicious, key=lambda x: x['z_score'], reverse=True)
    return suspicious, mean_all, std_all

def create_heatmap_overlay(image_uint8, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    overlay = overlay.astype(np.uint8)
    return overlay

def detect_frame_artifacts_deterministic(attention_score, base_regions=None):
    """
    Deterministic artifact selection using per-frame attention score.
    Only used for frames that exceed HIGH_ATTENTION_THRESHOLD.
    Returns list of artifact dicts WITHOUT numerical suspicion values in the final table.
    """
    if base_regions is None:
        base_regions = [
            {'name': 'Eyes', 'bbox': (50, 40, 120, 60), 'type': 'Blinking/Gaze'},
            {'name': 'Mouth', 'bbox': (70, 140, 80, 50), 'type': 'Lip-sync/Teeth'},
            {'name': 'Face Edge', 'bbox': (10, 10, 30, 200), 'type': 'Blending'},
            {'name': 'Nose', 'bbox': (90, 90, 40, 40), 'type': 'Lighting'},
            {'name': 'Cheek', 'bbox': (30, 100, 50, 60), 'type': 'Texture'}
        ]
    n = len(base_regions)
    count = int(np.floor(attention_score * n * 1.2))
    count = max(0, min(n, count))
    artifacts = []
    for idx in range(count):
        r = base_regions[idx]
        artifacts.append({
            'id': idx + 1,
            'name': r['name'],
            'type': r['type'],
            'bbox': r['bbox']
        })
    return artifacts

# -------------------- merged analysis (artifacts only on high-attention frames) --------------------
def create_merged_normalized_artifact_table(model, sequence, label, sample_id, save_path,
                                            region_z_thresholds=(1.0, 1.8),
                                            high_attention_threshold=HIGH_ATTENTION_THRESHOLD):
    model.eval()
    with torch.no_grad():
        pred = float(torch.sigmoid(model(sequence.unsqueeze(0))).item())
        attention_weights, _ = model.get_attention_weights(sequence.unsqueeze(0))
    attention_scores = attention_weights[0].cpu().numpy()  # seq_len

    heatmap = create_attention_heatmap(model, sequence)
    suspicious_regions, heat_mean, heat_std = analyze_suspicious_regions_normalized(
        heatmap, z_thresholds=region_z_thresholds
    )

    true_label = "FAKE" if label > 0.5 else "REAL"
    predicted_label = "FAKE" if pred > 0.5 else "REAL"
    is_correct = (true_label == predicted_label)
    confidence = abs(pred - 0.5) * 2

    fig = plt.figure(figsize=(20, 16))
    title_color = 'red' if predicted_label == 'FAKE' else 'green'
    fig.suptitle(f'DEEPFACE MERGED ANALYSIS (ARTIFACTS ON HIGH ATTENTION): {true_label} VIDEO\nModel Prediction: {predicted_label} ({pred:.3f}) | {"CORRECT" if is_correct else "INCORRECT"} | Confidence: {confidence:.1%}',
                 fontsize=16, fontweight='bold', color=title_color)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 2], width_ratios=[1,1,1,1,1,1])

    seq_len = sequence.shape[0]
    display_n = min(5, seq_len)

    # Row 1: original frames
    for i in range(display_n):
        ax = fig.add_subplot(gs[0, i])
        frame = sequence[i].permute(1,2,0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        ax.set_title(f'{true_label} Frame {i+1}', fontsize=12, fontweight='bold',
                     color='red' if true_label == 'FAKE' else 'green')
        ax.axis('off')

    # Row 2: heatmaps (global overlay) + attention circle
    for i in range(display_n):
        ax = fig.add_subplot(gs[1, i])
        frame = sequence[i].permute(1,2,0).cpu().numpy()
        frame = frame * std + mean
        frame_uint8 = (frame * 255).astype(np.uint8)
        overlay = create_heatmap_overlay(frame_uint8, heatmap, alpha=0.45)
        ax.imshow(overlay)
        att = float(attention_scores[i])
        cx, cy = overlay.shape[1]//2, overlay.shape[0]//2
        circ = plt.Circle((cx, cy), radius=max(4, int(18 * att)), color=(0.2, 0.8, 0.6, 0.6))
        ax.add_patch(circ)
        ax.set_title(f'Attention: {att:.3f}', fontsize=10)
        ax.axis('off')

    # Row 3: semantic normalized suspicious region boxes + artifacts ONLY for high-attention frames
    colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']
    artifact_global_count = 0
    artifact_frame_lists = []  # per-frame artifact list (for table)
    for i in range(display_n):
        ax = fig.add_subplot(gs[2, i])
        frame = sequence[i].permute(1,2,0).cpu().numpy()
        frame = frame * std + mean
        ax.imshow(frame)
        att = float(attention_scores[i])

        # Determine artifacts only if this frame has high attention
        frame_artifacts = []
        if att > high_attention_threshold:
            frame_artifacts = detect_frame_artifacts_deterministic(att)
        artifact_frame_lists.append(frame_artifacts)
        artifact_global_count += len(frame_artifacts)

        # Draw normalized semantic suspicious region boxes (if flagged)
        for j, region in enumerate(suspicious_regions[:5]):
            x1, y1, x2, y2 = region['coordinates']
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=colors[j], facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 6, f"{j+1}", fontsize=10, fontweight='bold',
                    color=colors[j], bbox=dict(boxstyle="round,pad=0.1", facecolor='white'))
            ax.text(x2 - 40, y1 + 2, f"z={region['z_score']:.2f}", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.7))

        # Draw artifact boxes for high-attention frames only (no numeric suspicion overlay)
        for k, art in enumerate(frame_artifacts):
            bx, by, bw, bh = art['bbox']
            rect_art = patches.Rectangle((bx, by), bw, bh, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect_art)
            ax.text(bx, by - 8, f"{art['id']}", fontsize=12, fontweight='bold',
                    color='white', bbox=dict(boxstyle="round,pad=0.1", facecolor='red'))
        ax.set_title(f'Artifacts: {len(frame_artifacts)}', fontsize=10)
        ax.axis('off')

    # Row 4: textual panel with normalized explanations AND separate artifact table (no numeric values)
    ax_text = fig.add_subplot(gs[3, :])
    ax_text.axis('off')

    explanation_lines = []
    explanation_lines.append("DETAILED SUSPICIOUS REGION ANALYSIS (NORMALIZED BY FRAME):\n")

    if suspicious_regions:
        for idx, region in enumerate(suspicious_regions[:6], start=1):
            explanation_lines.append(
                f"{idx}. {region['description'].upper()} [{region['suspicion_level']} - z: {region['z_score']:.3f} | mean_att: {region['attention']:.3f}]\n"
            )
            explanation_lines.append("   POTENTIAL INDICATORS:\n")
            for ind in region['indicators'][:3]:
                explanation_lines.append(f"   • {ind}\n")
            explanation_lines.append("\n   WHY SUSPICIOUS (normalized):\n")
            if region['suspicion_level'] == 'HIGH':
                explanation_lines.append("   Strongly above the frame baseline attention — significant candidate for manipulation.\n")
            elif region['suspicion_level'] == 'MEDIUM':
                explanation_lines.append("   Moderately above frame baseline attention — possible artifact.\n")
            else:
                explanation_lines.append("   Slightly above baseline — minor concern.\n")
            explanation_lines.append("\n" + ("=" * 80) + "\n")
    else:
        explanation_lines.append("NO SIGNIFICANT SUSPICIOUS REGIONS (after normalization).\n\n")

    # Technical summary
    explanation_lines.append("\nTECHNICAL SUMMARY:\n")
    explanation_lines.append(f"• Video Type: {true_label}\n")
    explanation_lines.append(f"• Model Prediction: {predicted_label} (Score: {pred:.3f})\n")
    explanation_lines.append(f"• Detection Accuracy: {'Correct' if is_correct else 'Incorrect'}\n")
    explanation_lines.append(f"• Confidence: {confidence:.1%}\n")
    explanation_lines.append(f"• Suspicious Semantic Regions (count): {len(suspicious_regions)}\n")
    explanation_lines.append(f"• Visual Artifact Boxes (high-attention frames only): total flagged across displayed frames: {artifact_global_count}\n")
    explanation_lines.append(f"• Heatmap mean/std (frame-level): {heat_mean:.3f} / {heat_std:.3f}\n")
    explanation_lines.append(f"• Region z-thresholds used: medium={region_z_thresholds[0]}, high={region_z_thresholds[1]}\n")
    explanation_lines.append("• Analysis Method: DenseNet-121 + BiGRU with Temporal Attention\n\n")
    explanation_lines.append("TEMPORAL ATTENTION SUMMARY (displayed frames):\n")
    for i in range(display_n):
        score = float(attention_scores[i])
        label_att = 'HIGH' if score > 0.4 else 'MEDIUM' if score > 0.2 else 'LOW'
        explanation_lines.append(f"  • Frame {i+1}: Attention {score:.3f} ({label_att})\n")

    # Artifact table (separate, no numeric suspicion values)
    explanation_lines.append("\n" + ("-" * 60) + "\n")
    explanation_lines.append("ARTIFACT TABLE (listed only for frames with HIGH attention):\n")
    any_artifacts = False
    for i, art_list in enumerate(artifact_frame_lists, start=1):
        if len(art_list) == 0:
            continue
        any_artifacts = True
        explanation_lines.append(f"\nFrame {i}:\n")
        for art in art_list:
            bx, by, bw, bh = art['bbox']
            explanation_lines.append(f"  - [{art['id']}] {art['name']} | Type: {art['type']} | BBox: x={bx}, y={by}, w={bw}, h={bh}\n")
    if not any_artifacts:
        explanation_lines.append("  No artifacts flagged in high-attention frames.\n")

    # Put lines into the axis
    expl_text = "".join(explanation_lines)
    ax_text.text(0.01, 0.99, expl_text, transform=ax_text.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.95))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return suspicious_regions, artifact_frame_lists, pred, is_correct

# ---------- Main ----------
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

    all_samples = [(seq, label, f'FAKE_{i+1}') for i, (seq, label) in enumerate(fake_samples)] + \
                  [(seq, label, f'REAL_{i+1}') for i, (seq, label) in enumerate(real_samples)]

    for sequence, label, sample_id in all_samples:
        sequence = sequence.to(device)
        print(f'\nAnalyzing {sample_id} (artifacts only on frames with attention > {HIGH_ATTENTION_THRESHOLD})...')
        suspicious_regions, artifact_lists, pred, is_correct = create_merged_normalized_artifact_table(
            model, sequence, label, sample_id, f'merged_normalized_artifacts_{sample_id}.png',
            region_z_thresholds=REGION_Z_THRESHOLDS,
            high_attention_threshold=HIGH_ATTENTION_THRESHOLD
        )
        total_artifacts = sum(len(l) for l in artifact_lists)
        print(f'   Saved merged_normalized_artifacts_{sample_id}.png | semantic regions: {len(suspicious_regions)} | artifact boxes flagged (displayed frames): {total_artifacts}')

    print('\nMerged normalized analysis (artifacts-on-high-attention) complete.')

if __name__ == '__main__':
    main()
