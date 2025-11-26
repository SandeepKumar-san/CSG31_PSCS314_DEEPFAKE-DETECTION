import os
import torch
import pickle
import cv2
import random
from collections import defaultdict
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision import transforms
from cpu_optimized_config import setup_cpu_environment, CPU_CONFIG

def extract_base_name(filename):
    """Extract base name from video filename"""
    # Remove .mp4 extension
    name = filename.replace('.mp4', '')
    
    # For manipulated videos: 01_02__exit_phone_room__YVGY8LOK -> 01__exit_phone_room
    if '__' in name and len(name.split('__')) >= 2:
        parts = name.split('__')
        if '_' in parts[0] and len(parts[0].split('_')) == 2:
            # This is manipulated: 01_02 -> 01
            person_id = parts[0].split('_')[0]
            scene_name = parts[1]
            return f"{person_id}__{scene_name}"
        else:
            # This is original: 01__exit_phone_room
            return name
    
    return name

def group_videos_by_original():
    """Group manipulated videos by their original video"""
    original_path = "S:/total DFD data/DFD_original sequences"
    manipulated_path = "S:/total DFD data/DFD_manipulated_sequences"
    
    # Get all original videos
    original_videos = {}
    if os.path.exists(original_path):
        for video in os.listdir(original_path):
            if video.endswith('.mp4'):
                base_name = extract_base_name(video)
                original_videos[base_name] = os.path.join(original_path, video)
    
    # Group manipulated videos by original
    manipulated_groups = defaultdict(list)
    if os.path.exists(manipulated_path):
        for video in os.listdir(manipulated_path):
            if video.endswith('.mp4'):
                base_name = extract_base_name(video)
                if base_name in original_videos:
                    manipulated_groups[base_name].append(os.path.join(manipulated_path, video))
    
    print(f"Found {len(original_videos)} original videos")
    print(f"Found manipulated groups for {len(manipulated_groups)} originals")
    
    return original_videos, manipulated_groups

def create_balanced_dataset():
    """Create balanced dataset: 1 original + 1 manipulated per video group"""
    original_videos, manipulated_groups = group_videos_by_original()
    
    balanced_pairs = []
    
    for base_name, original_path in original_videos.items():
        if base_name in manipulated_groups and manipulated_groups[base_name]:
            # Randomly select 1 manipulated video for this original
            selected_manipulated = random.choice(manipulated_groups[base_name])
            
            balanced_pairs.append({
                'original': original_path,
                'manipulated': selected_manipulated,
                'base_name': base_name
            })
    
    print(f"Created {len(balanced_pairs)} balanced video pairs")
    return balanced_pairs

def preprocess_balanced_faces():
    """Preprocess faces with balanced original-manipulated pairs"""
    setup_cpu_environment()
    
    # Create balanced dataset
    balanced_pairs = create_balanced_dataset()
    
    # Limit to desired number of pairs
    max_pairs = min(len(balanced_pairs), 300)  # Max 300 pairs = 600 videos
    selected_pairs = random.sample(balanced_pairs, max_pairs)
    
    cache_dir = "s:/Capstone/Capstone/cached_faces"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize MTCNN
    mtcnn = MTCNN(image_size=224, margin=0, device='cpu', post_process=False)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    def process_video(video_path, label):
        """Extract and cache faces from single video"""
        cap = cv2.VideoCapture(video_path)
        faces = []
        
        frame_count = 0
        while len(faces) < CPU_CONFIG['sequence_length'] and frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            try:
                face = mtcnn(frame_rgb)
                if face is not None:
                    face_tensor = transform(face.permute(1, 2, 0).numpy().astype('uint8'))
                    faces.append(face_tensor)
            except:
                pass
                
            frame_count += 1
        
        cap.release()
        
        if len(faces) == 0:
            return None
        while len(faces) < CPU_CONFIG['sequence_length']:
            faces.append(faces[-1])
        faces = faces[:CPU_CONFIG['sequence_length']]
        
        return torch.stack(faces), label
    
    # Process balanced pairs
    cached_data = []
    
    for pair in tqdm(selected_pairs, desc="Processing balanced video pairs"):
        # Process original video (label 0)
        result = process_video(pair['original'], 0)
        if result:
            cached_data.append(result)
        
        # Process manipulated video (label 1)
        result = process_video(pair['manipulated'], 1)
        if result:
            cached_data.append(result)
    
    # Shuffle the data
    random.shuffle(cached_data)
    
    # Save cache
    cache_file = os.path.join(cache_dir, "balanced_preprocessed_faces.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
    
    print(f"✅ Cached {len(cached_data)} videos to {cache_file}")
    print(f"   Original videos: {len(cached_data)//2}")
    print(f"   Manipulated videos: {len(cached_data)//2}")
    print(f"   Cache size: {os.path.getsize(cache_file) / (1024*1024):.1f} MB")

if __name__ == "__main__":
    preprocess_balanced_faces()