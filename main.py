import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import math
import PeronaMalik as PM
import Wei
import Utilities as utils

ORIGINAL_IMAGE = 'barbara.png'
IMAGE_GAUSSIAN = 'results/barbara_gaussian.jpg'
IMAGE_SP = 'results/barabara_SP.jpg'

print('\t Beginning Execution \n')

image = np.array(Image.open(ORIGINAL_IMAGE).convert('L'))
gaussian_image = utils.noise_addition("gaussian", image)
salt_pepper_image = utils.noise_addition("saltpepper", image)

print('\t Plotting images \n')

figure = plt.figure()
plt.gray()

ax1 = figure.add_subplot(131)
plt.title("Original Image")
plt.axis('off')
ax2 = figure.add_subplot(132)
plt.title("Gaussian Noise Image")
plt.axis('off')
ax3 = figure.add_subplot(133)
plt.title("Salt & Pepper Noise Image")
plt.axis('off')

ax1.imshow(image)
ax2.imshow(gaussian_image)
ax3.imshow(salt_pepper_image)

figure.tight_layout()
plt.show()

imageio.imwrite(IMAGE_GAUSSIAN, np.uint8(gaussian_image * 255))
print("\tImage with Gaussian noise saved in: " + IMAGE_GAUSSIAN)
imageio.imwrite(IMAGE_SP, np.uint8(salt_pepper_image * 255))
print("\tImage with Salt & Pepper noise saved in: " + IMAGE_SP)

print('\t Starting Gaussian Blur \n')
utils.gaussian_blur_edges(IMAGE_GAUSSIAN, 'gaussian')
utils.gaussian_blur_edges(IMAGE_SP, 'saltpepper')

print('\t Running Perona-Malik Anisotropic Diffusion \n')

PM.run_PM(ORIGINAL_IMAGE, IMAGE_GAUSSIAN, IMAGE_SP)

print('\t Comparing Different k on Perona-Malik Anisotropic Diffusion \n')

PM.compare_k(ORIGINAL_IMAGE, IMAGE_GAUSSIAN, IMAGE_SP)

print('\t Running Anisotropic Diffusion with Wei''s Diffusion Coefficient \n')

Wei.run_PDE(ORIGINAL_IMAGE, IMAGE_GAUSSIAN, IMAGE_SP)

