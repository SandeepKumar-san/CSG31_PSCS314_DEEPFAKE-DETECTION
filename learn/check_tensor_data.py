import pickle
import torch

# Load cached data
with open('s:/Capstone/Capstone/cached_faces/balanced_preprocessed_faces.pkl', 'rb') as f:
    cached_data = pickle.load(f)

print(f"Total cached videos: {len(cached_data)}")
print(f"Data type: {type(cached_data[0])}")

# Get first video sample
first_video = cached_data[0]
face_tensor, label = first_video

print(f"\nTensor shape: {face_tensor.shape}")
print(f"Label: {label} ({'REAL' if label == 0 else 'FAKE'})")

# Show actual tensor values from one pixel
print(f"\nSample pixel values from frame 0, position (100,100):")
print(f"Red channel:   {face_tensor[0, 0, 100, 100]:.6f}")
print(f"Green channel: {face_tensor[0, 1, 100, 100]:.6f}")
print(f"Blue channel:  {face_tensor[0, 2, 100, 100]:.6f}")

# Show range of values
print(f"\nTensor value ranges:")
print(f"Min value: {face_tensor.min():.6f}")
print(f"Max value: {face_tensor.max():.6f}")
print(f"Mean value: {face_tensor.mean():.6f}")

# Show a small 3x3 patch from one frame
print(f"\nSample 3x3 patch from Red channel, frame 0:")
patch = face_tensor[0, 0, 110:113, 110:113]
for row in patch:
    print([f"{val:.3f}" for val in row])