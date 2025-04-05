import numpy as np 
from scipy import fftpack
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import io
from medmnist import BloodMNIST
from noiser import MedMNISTWrapper, MultipleNoise
import os 

class Processor:
    def __init__(self):
        pass 

    def get_power_spectra(self, image):
        # Check if the image has 3 channels (RGB)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input must be an RGB image")
        
        power_spectra = []
        for channel in range(3):
            # Extract the channel
            channel_data = image[:, :, channel]
            
            # Apply 2D FFT
            fft_data = fftpack.fft2(channel_data)
            
            # Shift the zero frequency component to the center
            fft_shifted = fftpack.fftshift(fft_data)
            
            # Calculate the power spectrum (squared magnitude of the complex values)
            power = np.abs(fft_shifted) ** 2
            
            # Apply logarithmic scaling to enhance visualization
            power_log = np.log1p(power)  # log(1 + power) to avoid log(0)
            
            power_spectra.append(power_log)

        power_spectra = np.array(power_spectra)
        
        return power_spectra #(3,224,224)

    def convert_power_spectrum(self, images):
        '''
        Takes in a list of images numpy and returns a list of log power structrum
        '''
        # Check if the image has 3 channels (RGB)
        power_spectra = []
        for image in images:
            spectrum = self.get_power_spectra(image) #(3,224,224)
            power_spectra.append(spectrum)
        
        return power_spectra

    def visualize_multiple_power_spectra(self, power_spectra_list, titles=None):
        """
        Visualize multiple sets of power spectra from different images.
        
        Parameters:
        power_spectra_list: List of power spectra lists, where each inner list contains 
                            the power spectra for R, G, B channels of one image (3,224,224) shape 
        titles: List of titles for each row of visualizations (optional)
        figsize: Size of the figure to display
        
        Returns:
        List of normalized averaged power spectra
        """
        num_images = len(power_spectra_list)
        
        figsize = (15, 5*num_images)

        # Create figure and subplots
        fig, axes = plt.subplots(num_images, 1, figsize=figsize)
        
        # If only one image, make axes 2D for consistent indexing
        if num_images == 1:
            axes = np.expand_dims(axes, axis=0)
        
        # Create a custom colormap similar to the one in the paper (blue to yellow)
        # The paper seems to use a colormap that goes from dark blue to cyan to yellow
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0, 0, 0.5), (0, 0.5, 1), (0, 1, 1), (1, 1, 0)]  # Dark blue to cyan to yellow
        cmap_custom = LinearSegmentedColormap.from_list('blue_yellow', colors, N=256)
        
        avg_power_list = []
        
        # For each image's power spectra
        for i, power_spectra in enumerate(power_spectra_list):
            # Check if we have three power spectra per image
            if len(power_spectra) != 3:
                raise ValueError(f"Expected 3 power spectra for RGB channels in image {i}")
            
            # Row title if provided
            row_title = titles[i] if titles and i < len(titles) else f"Image {i+1}"
        
            # Average the power spectra for a combined visualization
            avg_power = np.mean(power_spectra, axis=0)
            
            # Normalize the averaged power spectrum
            avg_power_norm = (avg_power - np.min(avg_power)) / (np.max(avg_power) - np.min(avg_power))
            avg_power_list.append(avg_power_norm)

            
            # Plot the averaged power spectrum
            im = axes[i].imshow(avg_power_norm, cmap=cmap_custom, origin='lower', aspect='equal')
            axes[i].set_title(f"{row_title} - Average")
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            
            # Add colorbar for the averaged spectrum
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        
        return avg_power_list
    
    def save_spectra(self, power_list, save_dir='/Users/justin/Desktop/Everything/Code/agentic_images/files/spectra', folder_name="some_spectra"):
        # Create the full path
        full_path = os.path.join(save_dir, folder_name)
        os.makedirs(full_path, exist_ok=True)  # Avoid crash if it already exists
        
        for idx, arr in enumerate(power_list):
            plt.figure(figsize=(4, 4))
            plt.imshow(arr, aspect='auto', origin='lower', cmap='viridis')
            plt.axis('off')  # Optional: hide axes
            plt.tight_layout(pad=0)
            save_path = os.path.join(full_path, f"{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

if __name__ == "__main__":
    blood_ds = BloodMNIST(split="val", download=True, size=224)
    blood_ds = MedMNISTWrapper(blood_ds)
    first_five_images = []
    for i in range(5):
        image = blood_ds[i]
        first_five_images.append(image)

    noiser = MultipleNoise(blood_ds)
    noisy_images, psnrs, ssims = noiser.get_noisy_versions("gaussian", [0.2, 0.01, 0.04, 0.1], index=0)


    Proc = Processor()
    spectra = Proc.convert_power_spectrum(noisy_images)
    avg_spectra = Proc.visualize_multiple_power_spectra(spectra)
    Proc.save_spectra(avg_spectra, folder_name="closer")

    
    pass 