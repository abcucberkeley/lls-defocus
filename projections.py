import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tifffile as tiff

def create_projections(data):
    xy_projection = np.sum(data, axis=0)
    xz_projection = np.sum(data, axis=1)
    yz_projection = np.sum(data, axis=2)

    return xy_projection, xz_projection, yz_projection

def plot_projections(xy_projection, xz_projection, yz_projection):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(xy_projection, cmap='gray')
    axes[0].set_title('XY Projection')

    axes[1].imshow(xz_projection, cmap='gray', aspect='auto')
    axes[1].set_title('XZ Projection')

    axes[2].imshow(yz_projection.T, cmap='gray', aspect='auto')
    axes[2].set_title('YZ Projection')

    plt.show()

def main():
    # Replace 'your_file.tif' with the path to your 3D TIF file
    file_path = '/clusterfs/nvme/ethan/dataset/aberrations/33.tif'

    # Read the TIF file
    data = tiff.imread(file_path)
    print(data.size)

    # Create XY, XZ, and YZ projections
    xy_projection, xz_projection, yz_projection = create_projections(data)

    # Plot the projections
    plot_projections(xy_projection, xz_projection, yz_projection)

if __name__ == "__main__":
    main()