# timing_probe.py
import time, torch, os
from pathlib import Path
from model import DeepFakeDetector
from cached_dataloader import get_cached_dataloaders
from cpu_optimized_config import get_cpu_model_config

def time_model_only(model, seq_tensor, device):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(seq_tensor.to(device))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.perf_counter() - start

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = get_cpu_model_config()
    model = DeepFakeDetector(
        sequence_length=model_config['sequence_length'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    ).to(device)
    model.load_state_dict(torch.load("best_deepfake_detector.pth", map_location=device))
    model.eval()
    
    # Use cached data for timing
    _, _, test_loader = get_cached_dataloaders()
    model_times = []
    
    print(f"Testing on {device}")
    for i, (sequences, labels) in enumerate(test_loader):
        if i >= 10:  # Test 10 batches
            break
        
        for seq in sequences:
            model_t = time_model_only(model, seq.unsqueeze(0), device)
            model_times.append(model_t)
    
    if model_times:
        avg_time = sum(model_times) / len(model_times)
        print(f"Model inference avg: {avg_time:.3f}s per video")
        print(f"Tested {len(model_times)} videos")
    else:
        print("No timing data collected")
