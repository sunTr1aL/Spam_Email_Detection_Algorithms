#test code to verify GPU availability
import torch
print(torch.cuda.is_available())     # Should print: True
print(torch.cuda.get_device_name(0)) # Shows GPU name