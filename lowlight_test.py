import torch
import os
import requests

def lowlight(image_path):
    model_path = "snapshots/Epoch99.pth"

    # Download model if not exists
    if not os.path.exists(model_path):
        url = "https://drive.google.com/file/d/1vBSFwe7Zbvc9nQI-1-uBdcEEFKBVpUO8/view?usp=drive_link"
        os.makedirs("snapshots", exist_ok=True)
        print("Downloading model...")
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)
        print("Download complete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))
    ...
