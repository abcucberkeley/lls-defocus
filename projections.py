import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile as tiff
import re

def parameters():
    params = []
    count = 1
    for amp in [0.0,0.5,1.0]:
        for lls_offset in [0.0,0.5,1.0]:
            for z in range(3,15):
                if z != 4:
                    params.append([count, amp, lls_offset, z])
    return params


def create_projections(data):
    # 2D images representing the distribution of voxel intensities along an axis
    xy_projection = np.sum(data, axis=0)
    xz_projection = np.sum(data, axis=1)
    yz_projection = np.sum(data, axis=2)

    return xy_projection, xz_projection, yz_projection

def plot_projections(xy_projection, xz_projection, yz_projection, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(xy_projection, cmap='gray')
    axes[0].set_title('XY Projection')
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel("Y axis")

    axes[1].imshow(xz_projection, cmap='gray', aspect='auto')
    axes[1].set_title('XZ Projection')
    axes[1].set_xlabel('X axis')
    axes[1].set_ylabel("Z axis")

    axes[2].imshow(yz_projection.T, cmap='gray', aspect='auto')
    axes[2].set_title('YZ Projection')
    axes[2].set_xlabel('Z axis')
    axes[2].set_ylabel("Y axis")

    fig.suptitle(title)

    plt.show()

def main():
    params = parameters()

    # Replace 'your_file.tif' with the path to your 3D TIF file
    file_path = '/clusterfs/nvme/ethan/dataset/aberrations/91.tif'
    match = re.search(r'\d+', file_path)
    file_num = int(match.group()) - 1

    title = f'{file_num+1}.tif, Amplitude: {params[file_num][1]}, LLS Offset: {params[file_num][2]}, Zernike Mode: {params[file_num][3]}'

    # Read the TIF file
    data = tiff.imread(file_path)
    print(data.shape)

    # Create XY, XZ, and YZ projections
    xy_projection, xz_projection, yz_projection = create_projections(data)

    # Plot the projections
    plot_projections(xy_projection, xz_projection, yz_projection, title)

if __name__ == "__main__":
    main()