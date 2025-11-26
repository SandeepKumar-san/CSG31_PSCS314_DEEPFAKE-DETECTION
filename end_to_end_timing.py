# end_to_end_timing.py
import time, torch
from model import DeepFakeDetector
from facenet_pytorch import MTCNN
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms

from cpu_optimized_config import get_model_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_config = get_model_config()
model = DeepFakeDetector(
    sequence_length=model_config['sequence_length'],
    hidden_size=model_config['hidden_size'],
    num_layers=model_config['num_layers'],
    dropout=model_config['dropout']
).to(device)
model.load_state_dict(torch.load("best_deepfake_detector.pth", map_location=device))
model.eval()

mtcnn = MTCNN(keep_all=False, device=device)  # uses GPU if available
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Get videos from both directories
manipulated_path = Path("S:/total DFD data/DFD_manipulated_sequences")
original_path = Path("S:/total DFD data/DFD_original sequences")

video_paths = []
if manipulated_path.exists():
    video_paths.extend(list(manipulated_path.glob("*.mp4"))[:5])
if original_path.exists():
    video_paths.extend(list(original_path.glob("*.mp4"))[:5])

print(f"Found {len(video_paths)} videos for timing test")
e2e_times = []
model_times = []

for vp in video_paths:
    cap = cv2.VideoCapture(str(vp))
    frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # start end-to-end timer (decode + detect + model)
    t0 = time.perf_counter()

    # face detection on frames
    faces = []
    for f in frames:
        pil = f  # already RGB numpy
        result = mtcnn.detect(pil, landmarks=False)
        if len(result) == 2:
            box, prob = result
        else:
            box, prob, _ = result
        if box is not None and len(box)>0:
            x1,y1,x2,y2 = [int(v) for v in box[0]]
            crop = pil[y1:y2, x1:x2]
            crop = cv2.resize(crop, (224,224))
            faces.append(transform(crop))
        if len(faces) >= 5:
            break
    if len(faces) < 5:
        # pad with last frame
        while len(faces) < 5:
            faces.append(faces[-1])

    seq = torch.stack(faces)  # (5,3,224,224)
    seq = seq.unsqueeze(0).to(device)  # batch 1

    # model-only timing (for this video)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_model_start = time.perf_counter()
    with torch.no_grad():
        _ = model(seq)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_model_end = time.perf_counter()

    # end-to-end timing end
    t1 = time.perf_counter()

    e2e_times.append(t1 - t0)
    model_times.append(t_model_end - t_model_start)

if e2e_times and model_times:
    print(f"E2E mean: {np.mean(e2e_times):.3f}s ± {np.std(e2e_times):.3f}s")
    print(f"Model-only mean: {np.mean(model_times):.3f}s ± {np.std(model_times):.3f}s")
    print(f"Tested {len(video_paths)} videos")
else:
    print("No timing data collected - check video paths")
