import skimage as ski 
from skimage.util import random_noise 
import skimage.metrics as metrics 
import numpy as np 
import matplotlib.pyplot as plt
from medmnist import BloodMNIST
from utils import save_rgb_image
from torch.utils.data import Dataset


class MedMNISTWrapper(Dataset):
    def __init__(self, ds):
        self.ds = ds 
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        pil_image = self.ds[index][0]
        # convert to float numpy array 
        image = np.array(pil_image)
        convertToFloat = (image.dtype == np.uint8)
        if convertToFloat:
            image = ski.util.img_as_float(image)
        return image 


def noise(image, noise_type, var):
    '''
    Returns a noisy version of image (in float [0,1]) as noised by process noise_type
    
    image: (Nd.array) 
    '''
    assert noise_type in ['gaussian', 'poisson', 'speckle'], "Only start with these noise processes"

    convertToFloat = (image.dtype == np.uint8)
    if convertToFloat:
        image = ski.util.img_as_float(image)

    if noise_type == "gaussian" or noise_type == "speckle":
        return random_noise(image, mode=noise_type, var=var)
    
    return random_noise(image, mode=noise_type)


class MultipleNoise:
    def __init__(self, ds=None):
        if ds is not None:
            self.ds = ds 
        pass 

    def get_noisy_versions(self, noise_type, vars=None, image=None, index=None):
        '''
        Returns noisy versions of an image and the corresponding PSNR and SSIM
        '''
        assert image is not None or index is not None, "must get image somehow"
        if image is None:
            image = self.ds[index] 
        
        noisy_images, psnrs, ssims = [], [], []

        for var in vars:
            noised_version = noise(image, noise_type, var)
            noisy_images.append(noised_version)
            # calculate PSNR and SSIM
            psnr = metrics.peak_signal_noise_ratio(image, noised_version)
            ssim = metrics.structural_similarity(image, noised_version, channel_axis=2, data_range=1.0)
            psnrs.append(psnr)
            ssims.append(ssim)
        
        return noisy_images, psnrs, ssims
        
    def visualize_noisy(self, noisy_versions, psnrs, ssims, figsize=(15,15)):
        num_images = len(noisy_versions)
        
        plt.figure(figsize=figsize)
        
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            image = noisy_versions[i]
            
            # Check if image is grayscale or color
            if len(image.shape) == 2:  # grayscale
                plt.imshow(image, cmap='gray')
            else:
                plt.imshow(image)
            
            plt.title(f'SSIM: {ssims[i]:.4f}\nPSNR: {psnrs[i]:.2f} dB')
            plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # load the blood cell dataset
    blood_ds = BloodMNIST(split="val", download=True, size=224)
    blood_ds = MedMNISTWrapper(blood_ds)

    image = blood_ds[0]
    save_rgb_image(image, "files/first_blood.png")

    noiser = MultipleNoise(blood_ds)
    noisy_images, psnrs, ssims = noiser.get_noisy_versions("gaussian", [0.001, 0.1, 0.5, 1.0], index=0)

    noiser.visualize_noisy(noisy_images, psnrs, ssims)

    pass 
