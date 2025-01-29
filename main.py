from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
from skimage.filters import median
from skimage.morphology import ball

path = 'starry_night.jpg'
picture = load_image(path)

# Apply median filtering with a ball-shaped structuring element
clean_image = median(picture, ball(3))

# Perform edge detection
the_final = edge_detection(clean_image)

# Create a binary image by thresholding
binary_image = the_final < 100

# Convert the binary image to a PIL Image and save
binary = Image.fromarray((binary_image * 255).astype(np.uint8))  # Scale to 0-255 for saving
binary.save('mybinary.png')
