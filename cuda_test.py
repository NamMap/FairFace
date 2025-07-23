import torch

# Check if CUDA (GPU) is available
print("Is CUDA available? ", torch.cuda.is_available())

# If yes, show GPU name and device info
if torch.cuda.is_available():
    print("GPU Name: ", torch.cuda.get_device_name(0))
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("Current CUDA Device: ", torch.cuda.current_device())
else:
    print("CUDA is not available. PyTorch is using CPU.")