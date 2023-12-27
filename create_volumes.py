import sys
sys.path.append('.')
# sys.path.append('./submod/opticalaberrations/src')
sys.path.append('/clusterfs/fiona/ethan/git-manage/opticalaberrations/src')

import warnings
warnings.filterwarnings("ignore")

# from submod.opticalaberrations.src import psf_dataset
# from submod.opticalaberrations.src.synthetic import SyntheticPSF
# from submod.opticalaberrations.src.wavefront import Wavefront

import psf_dataset
from synthetic import SyntheticPSF
from wavefront import Wavefront
from pathlib import Path
import numpy as np

# data_dir = "/Users/ethantam/Desktop/abc/lls-defocus/data"
# data_dir = "/clusterfs/nvme/ethan/dataset"
data_dir = "/clusterfs/nvme/ethan/lls-defocus/data"

def aberrated_defocused_psf(amp, lls_offset, zernike_mode, fourier_emb, name, photons):

    # set up PSF generator with correct voxel size
    gen = SyntheticPSF(
        psf_type='../lattice/YuMB_NAlattice0p35_NAAnnulusMax0p40_NAsigma0p1.mat',
        lam_detection=.510,
        psf_shape=[64, 64, 64],
        x_voxel_size=.108,
        y_voxel_size=.108,
        z_voxel_size=.2,
    )

    # define your own wavefront by manually setting the amplitude for any of the zernike modes
    zernikes = np.zeros(15)
    zernikes[zernike_mode] = amp

    phi = Wavefront(
        amplitudes=zernikes,
        distribution='powerlaw',
        mode_weights='pyramid',
        modes=15,
        signed=True,
        rotate=True,
        order='ansi',
        lam_detection=gen.lam_detection,
    )
        
    np.testing.assert_array_equal(phi.amplitudes, zernikes)

    # simulate a PSF
    sample = psf_dataset.simulate_psf(
        filename=name,
        outdir=Path(f"{data_dir}/test"),
        gen=gen,
        phi=phi,
        emb=fourier_emb,
        photons=photons,
        noise=True,                 # if you want to add read and shot noise to the PSF
        normalize=True,             # normalize the PSF by the max value
        lls_defocus_offset=lls_offset
    )

    assert sample.shape == gen.psf_shape

# for "aberrations" directory
# name = "1"
# fourier_emb = False
# photons = 100000
# for amp in [0.0,0.5,1.0]:
#     for lls_offset in [0.0,0.5,1.0]:
#         for zernike_mode in range(3,15):
#             if zernike_mode != 4:
#                 aberrated_defocused_psf(amp, lls_offset, zernike_mode, fourier_emb, name, photons)
#                 name = str(int(name) + 1)   


# for "test" directory
name = "1"
fourier_emb = False
lls_offset = 0
for photons in [100000, 300000, 500000]:
    for amp in [0.0, 0.1]:
        for zernike_mode in range(3,15):
            if zernike_mode != 4:
                aberrated_defocused_psf(amp, lls_offset, zernike_mode, fourier_emb, name, photons)
                name = str(int(name) + 1)   