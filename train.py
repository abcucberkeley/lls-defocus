import torch
from torch_geometric.loader import DataLoader
import json
import glob
import os
import skimage.io as io
import numpy as np

class PSFDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data = glob.glob(data_dir)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        photons = os.listdir(self.data_dir)
        for photon in photons:
            photons_path = os.path.join(self.data_dir, photon)
            photons_path = os.path.join(photons_path, "amp_p0-p0")
            defocused_psfs = os.listdir(photons_path)
            for defocused_psf in defocused_psfs:
                psf_path = os.path.join(photons_path, defocused_psf)
                files = os.listdir(psf_path)
                files = sorted(files) # sort file names
                for i in range(0, len(files) // 2, 2):
                    json_file, tif_file = files[i], files[i+1]
                    image = io.imread(tif_file)
                    with open(json_file,'r') as j :
                        sample_load_file=json.load(j)
                        lls_offset = sample_load_file["lls_defocus_offset"]
                        return lls_offset, image


train_path="/clusterfs/nvme/ethan/dataset/lls_defocus_only/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/mixed"

train_dataset = PSFDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, follow_batch=['x_s', 'x_t'])


# this should only return the dataloader for ONLY ONE defocus_xxx_xxx
def dataloader(path, batch_size, val_split):
    # path = the path inside one defocus_xxx_xxx
    data = sorted(os.listdir(path))
    num_data = len(data)
    num_val = int(num_data * val_split)
    num_train = num_data - num_val

    input_files = []
    gt_files = []
    for file in data:
        if file.endswith('.tif'):
            input_files.append(os.path.join(path, file))
        else:
            gt_files.append(os.path.join(path, file))
    input_files = np.array(input_files)
    gt_files = np.array(gt_files)

    idx = np.arange(num_data)
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    train_input_files = input_files[train_idx]
    train_gt_files = gt_files[train_idx]
    print('Training size: ', len(train_input_files))

    train_data = PSFDataset(train_input_files, train_gt_files)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    val_idx = idx[num_train:]
    val_input_filenames = input_files[val_idx]
    val_gt_filenames = gt_files[val_idx]
    





def train(input_path):
    photons = os.listdir(input_path)
    for photon in photons:
        photons_path = os.path.join(input_path, photon)
        photons_path = os.path.join(photons_path, "amp_p0-p0")
        defocused_psfs = os.listdir(photons_path)
        for defocused_psf in defocused_psfs:
            psf_path = os.path.join(photons_path, defocused_psf)
            files = os.listdir(psf_path)
            files = sorted(files) # sort file names
            for i in range(0, len(files) // 2, 2):
                json_file, tif_file = files[i], files[i+1]
                image = io.imread(tif_file)
                with open(json_file,'r') as j :
                    sample_load_file=json.load(j)
                    lls_offset = sample_load_file["lls_defocus_offset"]
                    return lls_offset, image






    for photons in glob.glob(train_path):
        photons_path = os.path.join(train_path, photons)
        photons_path = os.path.join(photons_path, "amp_p0-p0")
        train_dataset = PSFDataset(photons_path)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, follow_batch=['x_s', 'x_t'])
        for json, tif in train_loader:
            pass

    # train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=bsize, shuffle=True, pin_memory=pmem)

if __name__ == '__main__':
    pass