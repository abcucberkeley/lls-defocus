import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from matplotlib.colors import Normalize

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

def plot_projections(xy_projection, xz_projection, yz_projection, title, save_directory):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    print(xy_projection.shape)
    print(np.min(xy_projection), np.max(xy_projection))
    # norm_xy = Normalize(vmin=np.min(xy_projection), vmax=np.max(xy_projection))
    norm_xy = Normalize(vmin=0, vmax=1)
    print("Min value:", xy_projection.min())
    print("Max value:", xy_projection.max())

    axes[0].imshow(xy_projection, cmap='gray', norm=norm_xy)
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
    fig.savefig(save_directory)

def main():
    params = parameters()

    for i in range(1,2):
        
        file_path = f'/clusterfs/nvme/ethan/dataset/aberrations/{i}.tif'
        # match = re.search(r'\d+', file_path)
        # file_num = int(match.group()) - 1
        file_num = i - 1

        title = f'{file_num+1}.tif, Amplitude: {params[file_num][1]}, LLS Offset: {params[file_num][2]}, Zernike Mode: {params[file_num][3]}'

        # Read the TIF file
        data = tiff.imread(file_path)
        print(data.shape)

        # Create XY, XZ, and YZ projections
        xy_projection, xz_projection, yz_projection = create_projections(data)

        # Plot the projections
        save_directory = f"/clusterfs/nvme/ethan/dataset/plots/{i}.png"
        plot_projections(xy_projection, xz_projection, yz_projection, title, save_directory)

if __name__ == "__main__":
    main()