import torch
import torch.nn as nn
import torchvision.models as models

# Model will use automatic device detection from calling script

class DeepFakeDetector(nn.Module):
    def __init__(self, sequence_length=5, hidden_size=512, num_layers=2, dropout=0.3):
        super(DeepFakeDetector, self).__init__()
        
        # CNN Backbone: DenseNet-121 (automatic device detection)
        self.backbone = models.densenet121(pretrained=True)
        # Remove final classifier
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get feature dimension from DenseNet-121
        self.feature_dim = 1024
        
        # Temporal Head: Bidirectional GRU
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier Head: MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)  # Single logit output
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        
        # Reshape for CNN processing: (batch_size * seq_len, channels, height, width)
        x = x.view(-1, channels, height, width)
        
        # Extract features using DenseNet backbone
        features = self.feature_extractor(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        
        # Reshape back to sequence: (batch_size, seq_len, feature_dim)
        features = features.view(batch_size, seq_len, -1)
        
        # Process temporal sequence with GRU
        gru_out, _ = self.gru(features)
        
        # Use final time step output (concatenated forward and backward)
        final_output = gru_out[:, -1, :]
        
        # Classification
        logits = self.classifier(final_output)
        
        return logits.squeeze(-1)  # Remove last dimension for BCEWithLogitsLoss
    
    def get_feature_maps(self, x):
        """Extract feature maps from final DenseBlock for Grad-CAM"""
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        
        # Get features from final DenseBlock (before global pooling)
        features = self.backbone.features(x)
        return features