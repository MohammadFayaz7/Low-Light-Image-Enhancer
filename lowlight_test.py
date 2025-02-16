import torch
import torchvision
import model
import numpy as np
from PIL import Image
import torch.nn as nn
import os

# Function to load and process the image using pre-trained model
def lowlight(image_file):
    device = torch.device('cpu')  # Running on CPU

    # Load image using PIL
    data_lowlight = Image.open(image_file)

    # Preprocess the image
    data_lowlight = np.asarray(data_lowlight) / 255.0
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)  # Change channels to first dimension
    data_lowlight = data_lowlight.unsqueeze(0)  # Add batch dimension

    # Load the pre-trained model
    DCE_net = model.enhance_net_nopool().to(device)
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=device, weights_only=True))
  # Load pre-trained weights

    # Run the model to get enhanced image
    with torch.no_grad():
        _, enhanced_image, _ = DCE_net(data_lowlight.to(device))

    # Convert tensor back to PIL image and return it
    enhanced_image = enhanced_image.squeeze().cpu()  # Remove batch dimension and move to CPU
    enhanced_image = enhanced_image.permute(1, 2, 0)  # Convert to HWC format for saving
    enhanced_image = np.clip(enhanced_image.numpy() * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)
