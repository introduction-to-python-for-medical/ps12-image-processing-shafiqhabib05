from PIL import Image
import numpy as np
from scipy.signal import convolve2d

path = 'starry_night.jpg'

def load_image(path):
    """
    Load an image and convert it to a numpy array.
    
    Parameters:
        path (str): Path to the image file.
        
    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    image = Image.open(path)
    image = np.array(image)
    return image

def edge_detection(image):
    """
    Perform edge detection on an image using Sobel filters.
    
    Parameters:
        image (np.ndarray): Input image array (RGB).
        
    Returns:
        np.ndarray: Magnitude of edges detected in the image.
    """
    grey_night = np.mean(image, axis=2)  # Convert to grayscale by averaging channels
    
    # Sobel kernels
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Apply convolution
    edge_x = convolve2d(grey_night, kernel_x, mode='same', boundary='symm')
    edge_y = convolve2d(grey_night, kernel_y, mode='same', boundary='symm')
    
    # Compute edge magnitude
    edgeMAG = np.sqrt(edge_x*2 + edge_y*2)
    return edgeMAG
