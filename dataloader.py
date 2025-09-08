import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from mtcnn import MTCNN
import json
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
from PIL import Image

class DFDDataset(Dataset):
    def __init__(self, original_dir, manipulated_dir, sequence_length=5):
        self.original_dir = Path(original_dir)
        self.manipulated_dir = Path(manipulated_dir)
        self.sequence_length = sequence_length
        # Initialize MTCNN detector
        self.detector = MTCNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # ===== LIMITED DATASET PROCESSING (FIRST 5 VIDEOS ONLY) - START =====
        # This section processes only the first 5 videos from each directory for quick testing
        self.samples = []
        
        print(f"Looking for original videos in: {self.original_dir} (LIMITED TO 5)")
        print(f"Looking for manipulated videos in: {self.manipulated_dir} (LIMITED TO 5)")
        
        # Add original videos (REAL - label 0) - FIRST 5 ONLY
        if self.original_dir.exists():
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            original_count = 0
            for ext in video_extensions:
                for video_file in self.original_dir.glob(ext):
                    if original_count < 5:  # LIMIT TO 5 VIDEOS
                        self.samples.append({'path': str(video_file), 'label': 0})
                        original_count += 1
                    else:
                        break
                if original_count >= 5:
                    break
            print(f"Found {len([s for s in self.samples if s['label'] == 0])} original videos (limited to 5)")
        else:
            print(f"Original directory does not exist: {self.original_dir}")
        
        # Add manipulated videos (FAKE - label 1) - FIRST 5 ONLY
        if self.manipulated_dir.exists():
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
            manipulated_count = 0
            for ext in video_extensions:
                for video_file in self.manipulated_dir.glob(ext):
                    if manipulated_count < 5:  # LIMIT TO 5 VIDEOS
                        self.samples.append({'path': str(video_file), 'label': 1})
                        manipulated_count += 1
                    else:
                        break
                if manipulated_count >= 5:
                    break
            print(f"Found {len([s for s in self.samples if s['label'] == 1])} manipulated videos (limited to 5)")
        else:
            print(f"Manipulated directory does not exist: {self.manipulated_dir}")
        
        print(f"Total samples found: {len(self.samples)} (LIMITED DATASET FOR TESTING)")
        # ===== LIMITED DATASET PROCESSING (FIRST 5 VIDEOS ONLY) - END =====
        
        # ===== ORIGINAL FULL DATASET PROCESSING CODE (COMMENTED OUT) - START =====
        # # Collect samples from both directories
        # self.samples = []
        # 
        # print(f"Looking for original videos in: {self.original_dir}")
        # print(f"Looking for manipulated videos in: {self.manipulated_dir}")
        # 
        # # Add original videos (REAL - label 0)
        # if self.original_dir.exists():
        #     video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        #     for ext in video_extensions:
        #         for video_file in self.original_dir.glob(ext):
        #             self.samples.append({'path': str(video_file), 'label': 0})
        #     print(f"Found {len([s for s in self.samples if s['label'] == 0])} original videos")
        # else:
        #     print(f"Original directory does not exist: {self.original_dir}")
        # 
        # # Add manipulated videos (FAKE - label 1)
        # if self.manipulated_dir.exists():
        #     video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        #     for ext in video_extensions:
        #         for video_file in self.manipulated_dir.glob(ext):
        #             self.samples.append({'path': str(video_file), 'label': 1})
        #     print(f"Found {len([s for s in self.samples if s['label'] == 1])} manipulated videos")
        # else:
        #     print(f"Manipulated directory does not exist: {self.manipulated_dir}")
        # 
        # print(f"Total samples found: {len(self.samples)}")
        # ===== ORIGINAL FULL DATASET PROCESSING CODE (COMMENTED OUT) - END =====
        
        if len(self.samples) == 0:
            raise ValueError(f"No video files found in directories:\n- {self.original_dir}\n- {self.manipulated_dir}\nPlease check if directories exist and contain video files.")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def extract_frames(self, video_path, max_frames=30):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    def align_face(self, face_img, landmarks):
        # Reference points for alignment (normalized coordinates)
        ref_points = np.array([
            [0.31556875, 0.4615741],  # Left eye
            [0.68262291, 0.4615741],  # Right eye
            [0.50026249, 0.6405741],  # Nose
            [0.34947187, 0.8246741],  # Left mouth
            [0.65343127, 0.8246741]   # Right mouth
        ]) * 224
        
        # Convert landmarks tensor to numpy array
        if torch.is_tensor(landmarks):
            landmarks = landmarks.cpu().numpy()
        landmarks = np.array(landmarks).reshape(5, 2)
        
        # Get similarity transformation matrix
        tform = cv2.estimateAffinePartial2D(landmarks, ref_points)[0]
        
        if tform is not None:
            # Apply transformation
            aligned = cv2.warpAffine(face_img, tform, (224, 224))
            return aligned
        else:
            # Fallback: simple resize if transformation fails
            return cv2.resize(face_img, (224, 224))
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = self.extract_frames(sample['path'])
        
        aligned_faces = []
        for frame in frames:
            try:
                # Convert frame to PIL Image for MTCNN
                pil_frame = Image.fromarray(frame)
                
                # Detect faces and landmarks
                boxes, probs, landmarks = self.detector.detect(pil_frame, landmarks=True)
                
                if boxes is not None and len(boxes) > 0:
                    # Get highest confidence detection
                    best_idx = torch.argmax(probs).item()
                    box = boxes[best_idx]
                    landmark = landmarks[best_idx]
                    
                    # Extract face region (box format: [x1, y1, x2, y2])
                    x1, y1, x2, y2 = box.astype(int)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    if face_crop.size > 0:
                        # Align face using landmarks
                        aligned_face = self.align_face(face_crop, landmark)
                        aligned_faces.append(aligned_face)
                        
                        if len(aligned_faces) == self.sequence_length:
                            break
            except Exception as e:
                continue
        
        # Pad sequence if needed
        while len(aligned_faces) < self.sequence_length:
            if aligned_faces:
                aligned_faces.append(aligned_faces[-1])
            else:
                aligned_faces.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Transform to tensors
        sequence = torch.stack([self.transform(face) for face in aligned_faces[:self.sequence_length]])
        
        return sequence, torch.tensor(sample['label'], dtype=torch.float32)

# ===== LIMITED DATALOADER FUNCTION (FOR 5 VIDEOS TESTING) - START =====
# This function creates dataloaders with only first 5 videos from each directory
def create_dataloaders(original_dir, manipulated_dir, batch_size=2, train_split=0.6, val_split=0.2):
    dataset = DFDDataset(original_dir, manipulated_dir)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Please check your video directories.")
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader
# ===== LIMITED DATALOADER FUNCTION (FOR 5 VIDEOS TESTING) - END =====

# ===== ORIGINAL FULL DATALOADER FUNCTION (COMMENTED OUT) - START =====
# This is the original function that processes all videos in the directories
# Uncomment this when you want to train on the full dataset
# def create_dataloaders(original_dir, manipulated_dir, batch_size=8, train_split=0.7, val_split=0.15):
#     dataset = DFDDataset(original_dir, manipulated_dir)
#     
#     if len(dataset) == 0:
#         raise ValueError("Dataset is empty. Please check your video directories.")
#     
#     # Split dataset
#     total_size = len(dataset)
#     train_size = int(train_split * total_size)
#     val_size = int(val_split * total_size)
#     test_size = total_size - train_size - val_size
#     
#     train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size, test_size]
#     )
#     
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
#     
#     return train_loader, val_loader, test_loader
# ===== ORIGINAL FULL DATALOADER FUNCTION (COMMENTED OUT) - END =====