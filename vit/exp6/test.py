import time
start = time.time()
import torch
torch_import = time.time()

print("Versione pytorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
torch_exec = time.time()

print("Pytorch import time", torch_import - start)
print("Pytorch exec", torch_exec - torch_import)

