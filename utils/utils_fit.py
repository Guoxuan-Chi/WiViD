import os
import torch
from utils.utils import get_lr


def fit_one_epoch(diffusion_model, optimizer,
                epoch, train_dataset, dataloader, Epoch, device, save_period, save_dir):
    total_loss = 0

    for depth, depth_valid, rgb, csi in dataloader:
        with torch.no_grad():
            depth = depth.to(device)
            depth_valid = depth_valid.to(device)
            rgb = rgb.to(device)
            csi = csi.to(device)
        
        optimizer.zero_grad()
        diffusion_loss = torch.mean(diffusion_model(x = depth, rgb_condition = rgb, csi_condition = csi, valid = depth_valid))
        diffusion_loss.backward()
        optimizer.step()

        total_loss += diffusion_loss.item() * depth.size(0)
 
    total_loss = total_loss / train_dataset.__len__()

    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch), 'loss: %.4f ' % (total_loss), 'lr: ', get_lr(optimizer))

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, 'Epoch%d.pth'%(epoch + 1)))

    torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
