import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

from model import DeepFakeDetector
from dataloader import DFDDataset

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks (using full backward hook to avoid deprecation warning)
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, class_idx=None):
        # Forward pass
        output = self.model(input_tensor)
        
        # For binary classification, use the single output logit
        if class_idx is None:
            # Use the raw logit for binary classification
            target_output = output[0] if output.dim() > 0 else output
        else:
            target_output = output[0]
        
        # Backward pass
        self.model.zero_grad()
        target_output.backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # First frame
        activations = self.activations[0]  # First frame
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlayed

def visualize_sequence_gradcam(model, sequence, labels, save_path='gradcam_visualization.png'):
    """Generate Grad-CAM visualization for a sequence"""
    model.eval()
    
    # Get target layer (final DenseBlock) - use a more accessible layer
    try:
        target_layer = model.backbone.features.denseblock4.denselayer16.conv2
    except AttributeError:
        # Fallback to final convolutional layer
        target_layer = model.backbone.features[-1]
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate CAM for the sequence
    sequence_tensor = sequence.unsqueeze(0)  # Add batch dimension
    cam = gradcam.generate_cam(sequence_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Denormalize images for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(5):
        # Original frame
        frame = sequence[i].permute(1, 2, 0).cpu().numpy()
        frame = frame * std + mean
        frame = np.clip(frame, 0, 1)
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Frame {i+1}')
        axes[0, i].axis('off')
        
        # Grad-CAM overlay (using first frame's CAM for all frames as example)
        overlayed = overlay_heatmap(frame_uint8, cam)
        axes[1, i].imshow(overlayed)
        axes[1, i].set_title(f'Grad-CAM {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'DeepFake Detection Explanation (Label: {"FAKE" if labels.item() > 0.5 else "REAL"})', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Configuration
    original_dir = './data/DFD_original sequences'
    manipulated_dir = './data/DFD_manipulated_sequences'
    model_path = 'best_model.pth'
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.is_available():
        print('GPU available but will use automatic detection')
        torch.cuda.empty_cache()
    else:
        print('Using CPU for explanation')
    
    # Load model
    print('Loading trained model...')
    model = DeepFakeDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Load dataset
    print('Loading dataset...')
    dataset = DFDDataset(original_dir, manipulated_dir)
    
    # Find a fake sample for explanation
    fake_indices = [i for i, sample in enumerate(dataset.samples) if sample['label'] == 1]
    
    if not fake_indices:
        print("No fake samples found in dataset!")
        return
    
    # Get a fake sample
    sample_idx = fake_indices[0]
    sequence, label = dataset[sample_idx]
    sequence = sequence.to(device)
    
    print(f'Explaining sample {sample_idx} (Label: {"FAKE" if label > 0.5 else "REAL"})')
    
    # Generate explanation
    visualize_sequence_gradcam(model, sequence, label)
    
    print('Grad-CAM visualization saved as gradcam_visualization.png')

if __name__ == '__main__':
    main()