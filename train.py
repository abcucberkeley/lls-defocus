import torch
from torch_geometric.loader import DataLoader
import json
import glob
import os

class PSFDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data = glob.glob(data_dir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        for photons in glob.glob(self.data_dir):
            photons_path = os.path.join(self.data_dir, photons)
            photons_path = os.path.join(photons_path, "amp_p0-p0")
            defocused_psfs = glob.glob(photons_path)
            for i in range(0, len(defocused_psfs) // 2, 2):
                json_file, tif_file = defocused_psfs[i], defocused_psfs[i+1]
                with open(json_file,'r') as j :
                    sample_load_file=json.load(j)
                    lls_offset = sample_load_file["lls_defocus_offset"].values()
                    return lls_offset, tif_file


train_path="/clusterfs/nvme/ethan/dataset/lls_defocus_only/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/mixed"
train_dataset = PSFDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, follow_batch=['x_s', 'x_t'])

def train():
    pass