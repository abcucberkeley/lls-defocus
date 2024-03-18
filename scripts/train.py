import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import os
import skimage.io as io
import numpy as np
from tqdm import tqdm
import csv
from loss_graph import plot_loss
import cli
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# separate dataloader, conv model, etc into different files later on
# watch nvidia-smi to check that ur gpu is being used or not

class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        # gelu instead of relu, less rigid activation func than relu

        # resnet??? suggested by sayan

        # batchnorm3d - ensures that model doesn't depend on features with higher values 
        #relu
        # dropout - ensures that no one node will contribute to  uch. tries to ensure convergence to reach global min

        # convolutional layers
        # [batch_size, channels, depth, height, width]
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # max pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)

        # dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # shape of x is (batch_size, 64, 64, 64)
        x = x.unsqueeze(1) # add dimension at index 1
        # shape of x is (batch_size, 1, 64, 64, 64)
        x = self.dropout(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # reshape into 2d tensor
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
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
        image = torch.from_numpy(image) # turn numpy array into tensor
        # print("Image shape", image.shape)
        with open(self.gt_files[idx],'r') as j:
            load_json = json.load(j)
            lls_offset = load_json["lls_defocus_offset"]
            lls_offset = round(lls_offset, 2) # round to to hundredths place
            return image.to(device), lls_offset

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
    num_train = int(num_data * val_split)
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


def train_no_amp(input_path, n_epochs, model_path, experiment_name):
    model = ConvModel()
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # create dataloader
    train_dataloader, val_dataloader = dataloader(input_path, batch_size=20, val_split=0.8) # increase batch_size to some value, try it out!

    # ensure that anything you want on GPU (like data) should be notated by .cuda() aka .to(device)

    # training - gpu
    # post-processing - cpu

    # create csv file
    with open(f'../experiments/{experiment_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        field = ["Training Loss", "Validation Loss"]
        writer.writerow(field)

    for epoch in tqdm(range(n_epochs)):
        train_total_loss = 0
        val_total_loss = 0

        # training
        for image, lls_offset in train_dataloader:
            # print("Image shape dataloader", image.shape)
            # result = model(image)
            # print(result.shape)
            # result = result.view(-1).to(torch.float64).to(device)
            # print(result.shape)
            lls_offset_pred = model(image).view(-1).to(torch.float64).to(device)
            loss = loss_fn(lls_offset_pred, lls_offset.to(device))
            train_total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        with torch.no_grad():
            model.eval()
            for image, lls_offset in val_dataloader:
                lls_offset_pred = model(image).view(-1).to(torch.float64).to(device)
                loss = loss_fn(lls_offset_pred, lls_offset.to(device)).cpu().detach().numpy()
                val_total_loss += loss

        print(f'Epoch: {epoch}, Training Loss: {train_total_loss / len(train_dataloader)}, Validation Loss: {val_total_loss / len(val_dataloader)}', flush=True)
        
        # from sayan, save every 10 epochs and best model best validation
        # save model at every 1000th epoch,
        # if epoch % 2 == 0 and epoch != 0:
        #     print("saving model")
        #     torch.save(model.state_dict(), model_path)

        # write to csv file
        with open(f'../experiments/{experiment_name}.csv', 'a', newline='') as f: # i changed from w to a, to append just check again
            writer = csv.writer(f)
            train_loss = (train_total_loss / len(train_dataloader)).item()
            val_loss = (val_total_loss / len(val_dataloader)).item()
            writer.writerow([train_loss, val_loss])
        
        # update loss graph
        # plot_loss(experiment_name, epoch)
    

def train(input_path, n_epochs):
    model = ConvModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(n_epochs)):
        photons = os.listdir(input_path)
        for photon in photons:
            photons_path = os.path.join(input_path, photon)
            photons_path = os.path.join(photons_path, "amp_p0-p0")
            defocused_psfs = os.listdir(photons_path)

            for defocused_psf in defocused_psfs:
                train_total_loss = 0
                val_total_loss = 0
                psf_path = os.path.join(photons_path, defocused_psf)
                train_dataloader, val_dataloader = dataloader(psf_path, batch_size=1, val_split=0.8)
                
                # training
                for image, lls_offset in train_dataloader:
                    lls_offset_pred = model(image).view(-1).to(torch.float64)
                    loss = loss_fn(lls_offset_pred, lls_offset)
                    train_total_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                # validation
                for image, lls_offset in val_dataloader:
                    lls_offset_pred = model(image).view(-1).to(torch.float64)
                    loss = loss_fn(lls_offset_pred, lls_offset)
                    val_total_loss += loss

                print(f'Epoch: {epoch}, Training Loss: {train_total_loss / len(train_dataloader)}, Validation Loss: {val_total_loss / len(val_dataloader)}')

def parse_args(args):
    parser = cli.argparser()
    parser.add_argument("--input_path", type=str, default='1')
    parser.add_argument("--n_epochs", type=int, default='1')
    parser.add_argument("--model_path", type=str, default='1')
    parser.add_argument("--experiment_name", type=str, default='1')
    return parser.parse_args(args)

def main(args=None):
    args = parse_args(args)
    print(args.input_path)
    print(args.n_epochs)
    print(args.model_path)
    print(args.experiment_name)
    train_no_amp(args.input_path, args.n_epochs, args.model_path, args.experiment_name)
    

if __name__ == '__main__':
    print('test')
    main()
    #input_path="/clusterfs/nvme/ethan/dataset/lls_defocus_only/YuMB_lambda510/z200-y108-x108/z64-y64-x64/z15/mixed"
    #input_path = '/clusterfs/nvme/ethan/dataset/no_amplitude'
    # experiment_name = 'test-001'
    # input_path = '/clusterfs/nvme/ethan/dataset/no_amplitude_large'
    # model_path = "/clusterfs/nvme/ethan/lls-defocus/models"
    # n_epochs = 1000
    # train(input_path, n_epochs)
    # train_no_amp(input_path, n_epochs, model_path, experiment_name)