import torch
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.current_device()}")
else:
    print("No GPU.")