import torch
checkpoint = torch.load('RETFound_cfp_weights.pth', map_location='cpu')
checkpoint_model = checkpoint['model']
for name, tensor in checkpoint_model.items():
    print(name.startswith('decoder'), tensor.size())

