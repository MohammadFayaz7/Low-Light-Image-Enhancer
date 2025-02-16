import torch
import torch.nn as nn
import torchvision
import torch.optim
import os
import argparse
import dataloader
import model
import Myloss
import re

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def extract_epoch(checkpoint_name):
    match = re.search(r"Epoch(\d+)_", checkpoint_name)
    return int(match.group(1)) if match else -1  # Return -1 if no match is found

def train(config):
    device = torch.device("cpu")  # Use CPU for training
    DCE_net = model.enhance_net_nopool().to(device)  # Send model to CPU
    DCE_net.apply(weights_init)

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    start_epoch = 0  # Default starting epoch

    # Load checkpoint if available
    if os.path.exists(config.snapshots_folder):
        checkpoints = [f for f in os.listdir(config.snapshots_folder) if f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=extract_epoch)
            checkpoint_path = os.path.join(config.snapshots_folder, latest_checkpoint)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):  # Check if it's a full checkpoint
                DCE_net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint.get('epoch', extract_epoch(latest_checkpoint)) + 1  # Ensure correct resume
                print(f"Resuming training from {latest_checkpoint} at epoch {start_epoch}")
            else:
                DCE_net.load_state_dict(checkpoint)
                print(f"Loaded model weights from {latest_checkpoint}, but no optimizer state found.")

    # Load training data
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=False)

    # Loss functions
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    DCE_net.train()

    for epoch in range(start_epoch, config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.to(device)  # Send input to CPU

            # Forward pass
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            # Compute losses
            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            # Total loss
            loss = Loss_TV + loss_spa + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if (iteration + 1) % config.display_iter == 0:
                print(f"Epoch [{epoch+1}/{config.num_epochs}], Iteration [{iteration+1}/{len(train_loader)}], Loss: {loss.item()}")

            # Save model snapshot
            if (iteration + 1) % config.snapshot_iter == 0:
                os.makedirs(config.snapshots_folder, exist_ok=True)
                checkpoint_path = os.path.join(config.snapshots_folder, f"Epoch{epoch}_Iter{iteration+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': DCE_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        scheduler.step()  # Adjust learning rate at the end of each epoch
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")

    config = parser.parse_args()
    train(config)
