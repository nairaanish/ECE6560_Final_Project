import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.util import random_noise
from PIL import Image
import imageio
import math

GAUSSIAN_RESULT = 'results/barbara_gaussian_blur.jpg'
GAUSSIAN_EDGES = 'results/barbara_gaussian_edges.jpg'
SALT_AND_PEPPER_RESULT ='results/barbara_gaussian_blur_SP.jpg'
SALT_AND_PEPPER_EDGES = 'results/barbara_gaussian_edges_SP.jpg'

def noise_addition(noise, img):

    if noise == "gaussian":
        noisy_image = random_noise(img, 'gaussian', seed = None, clip = True)
        return noisy_image

    elif noise == 'saltpepper':
        noisy_image = random_noise(img, 's&p', seed = None, clip = True)
        return noisy_image

def edge_detection(img):

    vertical_edges = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.int32)
    horizontal_edges = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

    dx_image = ndimage.filters.convolve(img / 255, vertical_edges)
    dy_image = ndimage.filters.convolve(img / 255, horizontal_edges)
    image_edges = np.hypot(dx_image, dy_image)

    return image_edges

def gaussian_blur_edges(noise_img, noise):

    if noise == 'gaussian':

        print('\t Applying the Gaussian Blur \n')

        noisy_image = np.array(Image.open(noise_img).convert('L'))

        gauss_img = ndimage.gaussian_filter(noisy_image, sigma = 5)

        imageio.imwrite(GAUSSIAN_RESULT, gauss_img)

        print('\nEdge Detection')

        gauss_img_edges = edge_detection(gauss_img)

        imageio.imwrite(GAUSSIAN_EDGES, gauss_img_edges)

        figure = plt.figure()

        plt.gray()
        plt.axis('off')
        plt.title('Blur and edges with Gaussian noise')

        ax1 = figure.add_subplot(131)
        plt.title("Noise Image")
        plt.axis('off')

        ax2 = figure.add_subplot(132)
        plt.title('Filtered Image')
        plt.axis('off')

        ax3 = figure.add_subplot(133)
        plt.title('Edges')
        plt.axis('off')

        ax1.imshow(noisy_image)
        ax2.imshow(gauss_img)
        ax3.imshow(gauss_img_edges)

        figure.tight_layout()
        plt.show()
    
    else:

        print('\t Applying the Gaussian Blur \n')

        noisy_image = np.array(Image.open(noise_img).convert('L'))

        gauss_img = ndimage.gaussian_filter(noisy_image, sigma = 5)

        imageio.imwrite(SALT_AND_PEPPER_RESULT, gauss_img)

        print('\nEdge Detection')

        gauss_img_edges = edge_detection(gauss_img)

        imageio.imwrite(SALT_AND_PEPPER_EDGES, gauss_img_edges)

        figure = plt.figure()

        plt.gray()
        plt.axis('off')
        plt.title('Blur and edges with Salt and Pepper noise')

        ax1 = figure.add_subplot(131)
        plt.title("Noise Image")
        plt.axis('off')

        ax2 = figure.add_subplot(132)
        plt.title('Filtered Image')
        plt.axis('off')

        ax3 = figure.add_subplot(133)
        plt.title('Edges')
        plt.axis('off')

        ax1.imshow(noisy_image)
        ax2.imshow(gauss_img)
        ax3.imshow(gauss_img_edges)

        figure.tight_layout()
        plt.show()
        

def PSNR(img, ref):
    
    mse = np.mean((img - ref) ** 2)

    if mse == 0:
        return 100

    max_val = 1.0

    return 20 * math.log10(max_val / math.sqrt(mse))