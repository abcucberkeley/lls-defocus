import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import skimage.io as io
import numpy as np

class ConvModel(nn.Model):
    def __init__(self):
        super()
        pass
    def forward(self, x):
        pass
    
class PSFDataset(torch.utils.data.Dataset):
    # input files : list of volumes
    # gt_files : list of json files with the lls_defocus_offset value
    def __init__(self, input_files, gt_files):
        self.input_files = input_files
        self.gt_files = gt_files

    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        image = io.imread(self.input_files[idx])
        with open(self.gt_files[idx],'r') as j:
            load_json = json.load(j)
            lls_offset = load_json["lls_defocus_offset"]
            lls_offset = round(lls_offset, 2) # round to to hundredths place
            return image, lls_offset

# this should only return the dataloader for ONLY ONE defocus_xxx_xxx
def dataloader(path, batch_size, val_split):
    # path = the path inside one defocus_xxx_xxx
    data = sorted(os.listdir(path))
    input_files = []
    gt_files = []
    for file in data:
        if file.endswith('.tif'):
            input_files.append(os.path.join(path, file))
        else:
            gt_files.append(os.path.join(path, file))
    input_files = np.array(input_files)
    gt_files = np.array(gt_files)

    num_data = len(input_files)
    idx = np.arange(num_data)
    np.random.shuffle(idx)
    num_val = int(num_data * val_split)
    num_train = num_data - num_val

    # training set
    train_idx = idx[:num_train]
    train_input_filenames = input_files[train_idx]
    train_gt_filenames = gt_files[train_idx]
    print('Training size: ', len(train_input_filenames))

    train_data = PSFDataset(train_input_filenames, train_gt_filenames)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # validation set
    val_idx = idx[num_train:]
    val_input_filenames = input_files[val_idx]
    val_gt_filenames = gt_files[val_idx]
    print('Validation size: ', len(val_input_filenames))

    val_data = PSFDataset(val_input_filenames, val_gt_filenames)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader
    

def train(input_path, n_epochs):
    model = ConvModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(n_epochs):
        photons = os.listdir(input_path)
        for photon in photons:
            photons_path = os.path.join(input_path, photon)
            photons_path = os.path.join(photons_path, "amp_p0-p0")
            defocused_psfs = os.listdir(photons_path)

            for defocused_psf in defocused_psfs:
                train_total_loss = 0
                val_total_loss = 0
                psf_path = os.path.join(photons_path, defocused_psf)
                train_dataloader, val_dataloader = dataloader(psf_path, 1)
                
                # training
                for image, lls_offset in train_dataloader:
                    lls_offset_pred = model(image)
                    loss = loss_fn(lls_offset_pred, lls_offset)
                    train_total_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # validation
                abs_difference = 0
                for image, lls_offset in val_dataloader:
                    lls_offset_pred = model(image)
                    loss = loss_fn(lls_offset_pred, lls_offset)
                    abs_difference += abs(lls_offset_pred - lls_offset)
                    val_total_loss += loss

                print(f'Epoch: {epoch}, Training Loss: {train_total_loss / len(train_dataloader)}, Validation Loss: {val_total_loss / len(val_dataloader)}, Model Accuracy: {abs_difference / len(val_dataloader)}')

if __name__ == '__main__':
    input_path="/clusterfs/nvme/ethan/dataset/lls_defocus_only/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/mixed"
    n_epochs = 100
    train(input_path, n_epochs)