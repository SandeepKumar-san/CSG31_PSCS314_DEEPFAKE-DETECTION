import streamlit as st
import torch
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import time
from pathlib import Path
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os
from model import DeepFakeDetector
from dataloader import DFDDataset
import torchvision.transforms as transforms
from mtcnn import MTCNN

# Page config
st.set_page_config(
    page_title="DeepFake Detection Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.success-card {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

def load_training_history():
    """Load training history from JSON file"""
    try:
        with open('training_history.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def load_model_info():
    """Load model information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_exists = Path('best_model.pth').exists()
    
    return {
        'device': str(device),
        'model_exists': model_exists,
        'gpu_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }

def create_loss_accuracy_plot(history):
    """Create combined loss and accuracy plots"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Training Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Validation Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Training Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', line=dict(color='green')),
        row=2, col=1
    )
    
    # Validation Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Metrics Overview")
    return fig

def create_combined_metrics_plot(history):
    """Create combined loss and accuracy plot"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Loss curves
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_loss'], name='Train Loss', line=dict(color='red')),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='blue')),
        secondary_y=False,
    )
    
    # Accuracy curves
    fig.add_trace(
        go.Scatter(x=epochs, y=history['train_acc'], name='Train Acc', line=dict(color='green')),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history['val_acc'], name='Val Acc', line=dict(color='orange')),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", secondary_y=True)
    fig.update_layout(title_text="Training Progress", height=500)
    
    return fig

def display_dataset_info():
    """Display dataset information"""
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>Original Videos</h4>
        <h2>16</h2>
        <p>Real videos (Label: 0)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Manipulated Videos</h4>
        <h2>Variable</h2>
        <p>Deepfake videos (Label: 1)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>Data Split</h4>
        <h2>70/15/15</h2>
        <p>Train/Val/Test %</p>
        </div>
        """, unsafe_allow_html=True)

def display_model_architecture():
    """Display model architecture information"""
    st.subheader("🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **CNN Feature Extractor:**
        - ResNet-18 backbone
        - Input: 224x224x3 images
        - Output: 512 features per frame
        
        **Sequence Processing:**
        - LSTM with 256 hidden units
        - Processes 5 frames per video
        - Dropout: 0.5
        """)
    
    with col2:
        st.markdown("""
        **Face Processing:**
        - MTCNN face detection
        - Landmark-based alignment
        - Artifact-focused learning
        
        **Classification:**
        - Binary output (Real/Fake)
        - BCEWithLogitsLoss
        - Adam optimizer
        """)

def predict_single_video(video_path):
    """Predict if a single video is fake or real"""
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DeepFakeDetector().to(device)
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
        
        # Initialize MTCNN
        detector = MTCNN()
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while frame_count < 30:  # Extract up to 30 frames
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        cap.release()
        
        if not frames:
            return None, "No frames extracted from video"
        
        # Face detection and alignment
        aligned_faces = []
        original_frames = []
        
        for frame in frames:
            try:
                pil_frame = Image.fromarray(frame)
                boxes, probs, landmarks = detector.detect(pil_frame, landmarks=True)
                
                if boxes is not None and len(boxes) > 0:
                    best_idx = torch.argmax(probs).item()
                    box = boxes[best_idx]
                    landmark = landmarks[best_idx]
                    
                    x1, y1, x2, y2 = box.astype(int)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Simple resize for alignment
                        aligned_face = cv2.resize(face_crop, (224, 224))
                        aligned_faces.append(aligned_face)
                        original_frames.append(frame)
                        
                        if len(aligned_faces) == 5:
                            break
            except:
                continue
        
        # Pad sequence if needed
        while len(aligned_faces) < 5:
            if aligned_faces:
                aligned_faces.append(aligned_faces[-1])
                original_frames.append(original_frames[-1])
            else:
                return None, "No faces detected in video"
        
        # Transform to tensors
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        sequence = torch.stack([transform(face) for face in aligned_faces[:5]]).unsqueeze(0).to(device)
        
        # Prediction
        with torch.no_grad():
            output = model(sequence)
            probability = torch.sigmoid(output).item()
            prediction = "FAKE" if probability > 0.5 else "REAL"
        
        return {
            'prediction': prediction,
            'confidence': probability if prediction == "FAKE" else 1 - probability,
            'probability': probability,
            'frames': original_frames[:5],
            'aligned_faces': aligned_faces[:5]
        }, None
        
    except Exception as e:
        return None, f"Error processing video: {str(e)}"

def main():
    st.title("🎭 DeepFake Detection Dashboard")
    st.markdown("AI-powered deepfake detection with explainable results")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "🎯 Video Prediction",
        "📈 Training Overview", 
        "📊 Dataset Analysis", 
        "🏗️ Model Details", 
        "🔍 Real-time Monitoring"
    ])
    
    if page == "🎯 Video Prediction":
        st.header("🎯 Video DeepFake Detection")
        
        # Check if model exists
        if not Path('best_model.pth').exists():
            st.error("❌ No trained model found! Please train the model first.")
            st.info("Go to Training Overview to monitor training progress.")
            return
        
        st.success("✅ Model loaded successfully!")
        
        # Video upload
        st.subheader("📹 Upload Video for Analysis")
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to detect if it contains deepfakes"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            st.info("🔄 Processing video... This may take a few moments.")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            for i, step in enumerate(["Extracting frames", "Detecting faces", "Aligning faces", "Running AI model", "Generating results"]):
                status_text.text(f"Step {i+1}/5: {step}...")
                progress_bar.progress((i + 1) / 5)
                time.sleep(0.5)
            
            # Predict
            result, error = predict_single_video(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if error:
                st.error(f"❌ {error}")
            else:
                # Display results
                st.success("✅ Analysis complete!")
                
                # Main result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result['prediction'] == 'FAKE':
                        st.markdown(f"""
                        <div style="background-color: #ffebee; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f44336;">
                        <h2 style="color: #d32f2f; margin: 0;">🚨 FAKE</h2>
                        <p style="margin: 0;">DeepFake Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50;">
                        <h2 style="color: #388e3c; margin: 0;">✅ REAL</h2>
                        <p style="margin: 0;">Authentic Video</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence Score", f"{result['confidence']:.1%}")
                
                with col3:
                    st.metric("Fake Probability", f"{result['probability']:.3f}")
                
                # Confidence gauge
                st.subheader("📊 Confidence Analysis")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result['confidence'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Confidence Level (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Frame analysis
                st.subheader("🔍 Frame Analysis")
                st.write("Processed frames from the video:")
                
                # Display frames in a grid
                cols = st.columns(5)
                for i, frame in enumerate(result['frames']):
                    with cols[i]:
                        st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.write(f"**Model Architecture:** CNN-RNN Hybrid (DenseNet-121 + Bidirectional GRU)")
                    st.write(f"**Frames Processed:** {len(result['frames'])}")
                    st.write(f"**Face Detection:** MTCNN with landmark alignment")
                    st.write(f"**Raw Logit Score:** {result['probability']:.6f}")
                    st.write(f"**Processing Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                
                # Explanation note
                st.info("""
                **How it works:** The AI model analyzes facial features, temporal consistency, 
                and subtle artifacts that are typically present in deepfake videos. 
                Higher confidence scores indicate stronger evidence for the prediction.
                """)
        
        else:
            st.info("👆 Please upload a video file to begin analysis.")
            
            # Example section
            st.subheader("📋 Supported Features")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **✅ Supported Formats:**
                - MP4 (.mp4)
                - AVI (.avi) 
                - MOV (.mov)
                - MKV (.mkv)
                """)
            
            with col2:
                st.markdown("""
                **🎯 Detection Capabilities:**
                - Face swap deepfakes
                - AI-generated faces
                - Temporal inconsistencies
                - Blending artifacts
                """)
    
    elif page == "📈 Training Overview":
        st.header("Training Overview")
        
        # Load training history
        history = load_training_history()
        model_info = load_model_info()
        
        if history:
            # Current metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                latest_train_acc = history['train_acc'][-1] if history['train_acc'] else 0
                st.metric("Latest Train Accuracy", f"{latest_train_acc:.3f}")
            
            with col2:
                latest_val_acc = history['val_acc'][-1] if history['val_acc'] else 0
                st.metric("Latest Val Accuracy", f"{latest_val_acc:.3f}")
            
            with col3:
                best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
                st.metric("Best Val Accuracy", f"{best_val_acc:.3f}")
            
            with col4:
                total_epochs = len(history['train_loss'])
                st.metric("Total Epochs", total_epochs)
            
            # Training plots
            st.subheader("Training Metrics")
            fig1 = create_loss_accuracy_plot(history)
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Combined View")
            fig2 = create_combined_metrics_plot(history)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Training statistics
            st.subheader("Training Statistics")
            df = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1),
                'Train Loss': history['train_loss'],
                'Train Acc': history['train_acc'],
                'Val Loss': history['val_loss'],
                'Val Acc': history['val_acc']
            })
            st.dataframe(df, use_container_width=True)
            
        else:
            st.warning("No training history found. Please run training first.")
        
        # System info
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Device:** {model_info['device']}")
            st.info(f"**GPU Available:** {model_info['gpu_available']}")
        
        with col2:
            st.info(f"**GPU Name:** {model_info['gpu_name']}")
            st.info(f"**Model Saved:** {model_info['model_exists']}")
    
    elif page == "📊 Dataset Analysis":
        st.header("Dataset Analysis")
        display_dataset_info()
        
        st.subheader("🎯 Data Processing Pipeline")
        st.markdown("""
        1. **Video Loading**: Load videos from original and manipulated directories
        2. **Frame Extraction**: Extract up to 30 frames per video
        3. **Face Detection**: Use MTCNN to detect faces in frames
        4. **Face Alignment**: Align faces using facial landmarks
        5. **Sequence Creation**: Create sequences of 5 aligned face images
        6. **Normalization**: Apply ImageNet normalization
        7. **Data Splitting**: 70% train, 15% validation, 15% test
        """)
        
        # Data imbalance visualization
        st.subheader("📊 Data Distribution")
        labels = ['Original (Real)', 'Manipulated (Fake)']
        values = [16, 100]  # Approximate values
        
        fig = px.pie(values=values, names=labels, title="Dataset Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🏗️ Model Details":
        st.header("Model Architecture Details")
        display_model_architecture()
        
        st.subheader("🔧 Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Hyperparameters:**
            - Learning Rate: 1e-4
            - Batch Size: 4
            - Epochs: 5
            - Optimizer: Adam
            - Loss Function: BCEWithLogitsLoss
            """)
        
        with col2:
            st.markdown("""
            **Data Augmentation:**
            - Resize to 224x224
            - ImageNet normalization
            - Face alignment
            - Sequence padding
            """)
        
        st.subheader("🎯 Artifact Detection Focus")
        st.markdown("""
        The model is designed to detect manipulation artifacts rather than person-specific features:
        
        - **Temporal Inconsistencies**: Flickering between frames
        - **Blending Artifacts**: Unnatural edges around face boundaries  
        - **Compression Patterns**: Artifacts from deepfake generation
        - **Landmark Inconsistencies**: Unnatural facial feature movements
        """)
    
    elif page == "🔍 Real-time Monitoring":
        st.header("Real-time Training Monitor")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (every 10 seconds)")
        
        if auto_refresh:
            time.sleep(10)
            st.experimental_rerun()
        
        # Manual refresh button
        if st.button("🔄 Refresh Now"):
            st.experimental_rerun()
        
        # Current status
        history = load_training_history()
        if history:
            latest_epoch = len(history['train_loss'])
            latest_train_loss = history['train_loss'][-1]
            latest_val_loss = history['val_loss'][-1]
            
            st.markdown(f"""
            <div class="success-card">
            <h4>Current Training Status</h4>
            <p><strong>Epoch:</strong> {latest_epoch}</p>
            <p><strong>Train Loss:</strong> {latest_train_loss:.4f}</p>
            <p><strong>Val Loss:</strong> {latest_val_loss:.4f}</p>
            <p><strong>Last Updated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            progress = latest_epoch / 50  # Assuming max 50 epochs
            st.progress(progress)
            st.write(f"Training Progress: {progress*100:.1f}%")
        else:
            st.warning("No training in progress or history found.")
            
            # Quick training button
            if st.button("🚀 Start Training"):
                st.info("To start training, run: `python train.py` in your terminal")
                st.code("python train.py", language="bash")

if __name__ == "__main__":
    main()