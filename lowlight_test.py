import torch
import os
import requests

def download_from_google_drive(id, destination):
    """Download a file from Google Drive."""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)


def lowlight(image_path):
    model_path = "snapshots/Epoch99.pth"
    google_drive_id = "1vBSFwe7Zbvc9nQI-1-uBdcEEFKBVpUO8"  # extracted from your link

    if not os.path.exists(model_path):
        os.makedirs("snapshots", exist_ok=True)
        print("Downloading model...")
        download_from_google_drive(google_drive_id, model_path)
        print("Download complete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # You must define or import DCE_net before loading
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))
    DCE_net.to(device)
    DCE_net.eval()

    # Continue your inference process here
    # ...

