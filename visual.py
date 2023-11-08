from PIL import Image
def visualize_grad_cam(model, image, target_class, save_path):
    model.eval()

    # Hook to get the gradients and the feature maps
    gradients = None
    activations = None

    def hook_fn(module, input_grad, output_grad):
        nonlocal gradients
        gradients = output_grad[0]

    def hook_fn_features(module, input, output):
        nonlocal activations
        activations = output

    # Assuming you're using the last transformer block for visualization
    model.blocks[-1].register_backward_hook(hook_fn)
    model.blocks[-1].register_forward_hook(hook_fn_features)

    # Forward pass
    features = model.forward_features(image)
    logits = model.head(features)
    model.zero_grad()
    
    # Backward pass
    target = logits[0, target_class]
    target.backward()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    for i in range(activations.shape[1]):
        activations[:, i, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Upsample and save
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_image = heatmap_image.resize((image.shape[2], image.shape[3]))
    heatmap_image.save(save_path)
def visualize_grad_cam(model, image, target_class, save_path):
    model.eval()

    # Hook to get the gradients and the feature maps
    gradients = None
    activations = None

    def hook_fn(module, input_grad, output_grad):
        nonlocal gradients
        gradients = output_grad[0]

    def hook_fn_features(module, input, output):
        nonlocal activations
        activations = output

    # Assuming you're using the last transformer block for visualization
    model.blocks[-1].register_backward_hook(hook_fn)
    model.blocks[-1].register_forward_hook(hook_fn_features)

    # Forward pass
    features = model.forward_features(image)
    logits = model.head(features)
    model.zero_grad()
    
    # Backward pass
    target = logits[0, target_class]
    target.backward()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2])
    for i in range(activations.shape[1]):
        activations[:, i, :] *= pooled_gradients[i]
        
    heatmap = torch.mean(activations, dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Upsample and save
    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_image = heatmap_image.resize((image.shape[2], image.shape[3]))
    heatmap_image.save(save_path)

def visualize_attention(model, image, save_path, layer=-1, head=0):
    # Forward pass while storing attention weights
    B = image.shape[0]
    x = model.patch_embed(image)
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)
    
    attentions = []
    for blk in model.blocks:
        x, attn = blk(x, return_attention=True)
        attentions.append(attn)

    # Using the attention from the last layer
    attn = attentions[layer][0, head].detach().cpu().numpy()
    
    # Reshape to image size and save
    H, W = image.shape[2], image.shape[3]
    attn_image = Image.fromarray((attn * 255).astype(np.uint8)).resize((H, W))
    attn_image.save(save_path)
