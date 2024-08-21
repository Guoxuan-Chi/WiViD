import torch
import torch.nn
from gaussian_diffusion import GaussianDiffusion, generate_linear_schedule
from unet import UNet
from test_dataset import DiffusionDataset
from torch.utils.data import DataLoader
from utils.utils import postprocess_depth_output
import matplotlib.pyplot as plt
from indicator import *
import os

device = torch.device('cuda:0')

save_path = "PATH1"
test_log = save_path + "test_log.txt"
model_weight_path = "PATH2"

if not os.path.exists(save_path):
    os.mkdir(save_path)

num_timesteps   = 50
schedule_low    = 1e-4
schedule_high   = 0.02

betas = generate_linear_schedule(
        num_timesteps,
        schedule_low * 1000 / num_timesteps,
        schedule_high * 1000 / num_timesteps,
    )

model = GaussianDiffusion(UNet(), (160, 90), 1, betas).to(device)

model.load_state_dict(torch.load(model_weight_path, map_location=torch.device("cpu")))
torch.cuda.empty_cache()


test_dataset   = DiffusionDataset()
dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

print(test_dataset.__len__())

model.eval()
model = model.to(device)

i=0

absrel_sum = 0
sqrel_sum = 0
rmse_sum = 0
logrmse_sum = 0
tacc1_sum = 0
tacc2_sum = 0
tacc3_sum = 0
for depth, depth_valid, rgb, csi in dataloader:

    depth = depth.to(device)
    depth_valid = depth_valid.to(device)
    rgb = rgb.to(device)
    csi = csi.to(device)
    res_0 = model.sample(1, device, rgb, csi).to(device)
    res = postprocess_depth_output(res_0 * depth_valid).to(device)
    depth = postprocess_depth_output(depth)

    absrel, sqrel, rmse, logrmse, tacc1, tacc2, tacc3 = compute_errors(depth, res, depth_valid)

    absrel_sum += absrel
    sqrel_sum += sqrel
    rmse_sum += rmse
    logrmse_sum += logrmse
    tacc1_sum += tacc1
    tacc2_sum += tacc2
    tacc3_sum += tacc3
    print()

    save_depth = plt.imshow(depth[0].cpu().detach().data.numpy().transpose(2, 1, 0), cmap='viridis')
    plt.savefig(save_path+str(i)+'_depth.png')
    save_res = plt.imshow(res_0[0].cpu().detach().data.numpy().transpose(2, 1, 0), cmap='viridis')
    plt.savefig(save_path+str(i)+'_res.png')

    print("i: ", i, "AbsRel: ", absrel, "SqRel: ", sqrel, "RMSE: ", rmse, "LogRMSE: ", logrmse, "tacc1: ", tacc1)

    with open(test_log, 'a') as file:
        file.write(f"i: {i} AbsRel: {absrel} SqRel: {sqrel} RMSE: {rmse} LogRMSE: {logrmse} tacc1: {tacc1} tacc2: {tacc2} tacc3:{tacc3} \n")

    i+=1

absrel_sum /= test_dataset.__len__()
sqrel_sum /= test_dataset.__len__()
rmse_sum /= test_dataset.__len__()
logrmse_sum /= test_dataset.__len__()
tacc1_sum /= test_dataset.__len__()
tacc2_sum /= test_dataset.__len__()
tacc3_sum /= test_dataset.__len__()

print("AbsRel_sum: ", absrel_sum, "SqRel_sum: ", sqrel_sum, "RMSE_sum: ", rmse_sum, "LogRMSE_sum: ", logrmse_sum)

with open(test_log, 'a') as file:
    file.write(f"AbsRel_sum: {absrel_sum} SqRel_sum: {sqrel_sum} RMSE_sum: {rmse_sum} LogRMSE_sum: {logrmse_sum} tacc1_sum: {tacc1_sum} tacc2_sum: {tacc2_sum} tacc3_sum:{tacc3_sum} \n")
