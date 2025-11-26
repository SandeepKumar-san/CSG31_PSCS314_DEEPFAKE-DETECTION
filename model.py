import torch
import torch.nn as nn
import torchvision.models as models

# Model will use automatic device detection from calling script

class DeepFakeDetector(nn.Module):
    def __init__(self, sequence_length=5, hidden_size=512, num_layers=2, dropout=0.3):
        super(DeepFakeDetector, self).__init__()
        
        # CNN Backbone: DenseNet-121 optimized for deepfake detection
        self.backbone = models.densenet121(pretrained=True)
        
        # Extract features from DenseNet (remove classifier)
        self.feature_extractor = self.backbone.features
        
        # DenseNet-121 feature dimension
        self.feature_dim = 1024
        
        # Temporal Head: Bidirectional GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Enhanced Classifier Head for deepfake detection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # *2 for bidirectional
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, 1)  # Single logit output
        )
        
        # Global Average Pooling for DenseNet features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Initialize classifier weights
        self._initialize_classifier()
        
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
        
        return logits.view(-1)  # Ensure consistent [batch_size] shape
    
    def _initialize_classifier(self):
        """Initialize classifier weights for better convergence"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_feature_maps(self, x):
        """Extract feature maps from final DenseBlock for Grad-CAM"""
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        
        # Get features from DenseNet feature extractor
        features = self.feature_extractor(x)
        return features
    
    def freeze_backbone(self, freeze=True):
        """Freeze/unfreeze DenseNet backbone for fine-tuning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
    
    def get_attention_weights(self, x):
        """Get GRU attention weights for temporal analysis"""
        batch_size, seq_len, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)
        
        # Extract features
        features = self.feature_extractor(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        features = features.view(batch_size, seq_len, -1)
        
        # Get GRU outputs and hidden states
        gru_out, hidden = self.gru(features)
        
        # Compute attention weights (simplified)
        attention_weights = torch.softmax(torch.norm(gru_out, dim=2), dim=1)
        
        return attention_weights, gru_out