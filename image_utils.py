from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

path = 'brain-scan.jpg'  # Change to absolute path if necessary

def load_image(path):
    """Load an image and convert it to a numpy array."""
    image = Image.open(path).convert('RGB')  # Ensure it's RGB
    return np.array(image)

def edge_detection(image):
    """Perform edge detection on an image using Sobel filters."""
    gray_image = np.mean(image, axis=2).astype(np.float32)  # Convert to grayscale & float
    
    # Sobel kernels
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    # Apply convolution
    edge_x = convolve2d(gray_image, kernel_x, mode='same', boundary='symm')
    edge_y = convolve2d(gray_image, kernel_y, mode='same', boundary='symm')
    
    # Compute edge magnitude
    edgeMAG = np.sqrt(edge_x**2 + edge_y**2)
    
    return edgeMAG
