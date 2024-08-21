from torch.utils.data.dataset import Dataset
import torch

class DiffusionDataset(Dataset):
    def __init__(self):
        super(DiffusionDataset, self).__init__()
        self.test_data_path = "PATH"

    def __len__(self):
        return 1000
        
    def __getitem__(self, index):
        rgb_fast_path = self.fast_path("rgb", index)
        csi_fast_path = self.fast_path("csi", index)
        depth_fast_path = self.fast_path("depth", index)
        depth_valid_fast_path = self.fast_path("depth_valid", index)

        rgb = torch.load(rgb_fast_path)
        csi = torch.load(csi_fast_path)
        depth = torch.load(depth_fast_path)
        depth_valid = torch.load(depth_valid_fast_path)

        return depth, depth_valid, rgb, csi

    def fast_path(self, type, index):
        return self.test_data_path + type + "_" + str(index) + ".pt"
    



        
        
    