import imageio 
import numpy as np 
import os 

def save_rgb_image(image: np.ndarray, file_path: str):
    """
    Saves a 3-channel (RGB) NumPy image to a file path as a PNG.

    Parameters:
    - image: np.ndarray of shape (H, W, 3), dtype should be uint8 or float32/float64 in [0, 1]
    - file_path: str, where to save the image (should end in .png)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Normalize and convert if needed
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError("Unsupported image dtype. Use float32/64 in [0,1] or uint8.")

    # Save the image
    imageio.imwrite(file_path, image)