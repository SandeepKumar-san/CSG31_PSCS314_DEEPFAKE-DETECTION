# CSG31_PSCS314_DEEP-FAKE-DETECTION

## Explainable CNN-GRU Deepfake Detection with Grad-CAM Visual Interpretability

### 🎯 Project Overview
Advanced deepfake detection system using DenseNet-121 + Bidirectional GRU architecture with Grad-CAM explainable AI for transparent decision-making.

### 🏗️ Key Features
- **DenseNet-121**: Dense feature extraction (1024-dim vectors)
- **Bidirectional GRU**: Temporal sequence modeling
- **MTCNN**: Robust face detection and alignment
- **Grad-CAM**: Explainable AI with visual attention maps
- **Interactive Dashboard**: Streamlit web interface
- **Real-time Processing**: 2-3 seconds per video

### 🔬 Architecture Components
1. **Video Preprocessing**: Frame extraction and quality assessment
2. **Face Detection**: MTCNN-based detection and landmark alignment
3. **Feature Extraction**: DenseNet-121 spatial feature extraction
4. **Temporal Modeling**: Bidirectional GRU for sequence analysis
5. **Explainable AI**: Grad-CAM visualization for decision transparency
6. **Classification**: Binary real/fake prediction with confidence scores

### 🚀 Quick Start
```bash
# Setup and run experiments
python setup_experiments.py

# Train model
python train.py

# Evaluate performance
python evaluate.py

# Generate explanations
python explain.py

# Launch dashboard
python run_dashboard.py
```

### 📊 Performance
- **Accuracy**: ~89% on test dataset
- **Processing Time**: 2-3 seconds per video
- **Memory Usage**: <2GB GPU memory
- **XAI**: Visual attention maps for decision transparency

### 🔍 Explainable AI Features
- **Grad-CAM Visualization**: Highlights influential facial regions
- **Temporal Analysis**: Frame-by-frame attention tracking
- **Artifact Detection**: Focuses on blending and compression artifacts
- **Decision Transparency**: Visual evidence for model predictions
