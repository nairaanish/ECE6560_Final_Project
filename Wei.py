import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import Utilities as utils

def alpha(grad_I, k):

    return 2 - (2 / 1 + ((grad_I/k) ** 2))

def coeff_guo(I, k):

    return 1 / (1 + ((np.linalg.norm(I)/k) ** alpha(np.linalg.norm(I), k)))

def diffuse_new(img, log_f, iter, k, lmb = 0.01):

    image = np.array(Image.open(img).convert('L')) / 255
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = [image]

    for t in range(iter):
        I_Up = image[:-2, 1:-1] - image[1:-1, 1:-1]
        I_Down = image[2:, 1:-1] - image[1:-1, 1:-1]
        I_Right = image[1:-1, 2:] - image[1:-1, 1:-1]
        I_Left = image[1:-1, :-2] - image[1:-1, 1:-1]

        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lmb * (
            coeff_guo(I_Up, k) * I_Up +
            coeff_guo(I_Down, k) * I_Down +
            coeff_guo(I_Right, k) * I_Right +
            coeff_guo(I_Left, k) * I_Left
        )

        image = new_image

        if (t+1) % log_f == 0:
            result.append(image.copy())

    return result

def run_PDE(og_img, img_gauss, img_sp):
    
    k_gauss = 0.1
    iterations = 80
    num_col = 5

    pde_gauss = diffuse_new(img_gauss, log_f= iterations / (num_col - 1), iter=iterations, k=k_gauss, lmb=0.1)

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(221)
    plt.title("t = 0")
    plt.axis('off')
    ax1.imshow(pde_gauss[0])

    ax2 = figure.add_subplot(222)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_gauss[0]))

    ax3 = figure.add_subplot(223)
    plt.title("t = 20")
    plt.axis('off')
    ax3.imshow(pde_gauss[1])

    ax4 = figure.add_subplot(224)
    plt.axis('off')
    ax4.imshow(utils.edge_detection(pde_gauss[1]))

    plt.show()

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(221)
    plt.title("t = 40")
    plt.axis('off')
    ax1.imshow(pde_gauss[2])

    ax2 = figure.add_subplot(222)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_gauss[2]))

    ax3 = figure.add_subplot(223)
    plt.title("t = 60")
    plt.axis('off')
    ax3.imshow(pde_gauss[3])

    ax4 = figure.add_subplot(224)
    plt.axis('off')
    ax4.imshow(utils.edge_detection(pde_gauss[3]))

    plt.show()

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(121)
    plt.title("t = 80")
    plt.axis('off')
    ax1.imshow(pde_gauss[4])

    ax2 = figure.add_subplot(122)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_gauss[4]))

    plt.show()

    k_sp = 0.1
    iterations_sp = 160
    num_col = 5

    pde_sp = diffuse_new(img_sp, log_f= iterations_sp / (num_col - 1), iter=iterations_sp, k=k_sp, lmb=0.1)

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(221)
    plt.title("t = 0")
    plt.axis('off')
    ax1.imshow(pde_sp[0])

    ax2 = figure.add_subplot(222)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_sp[0]))

    ax3 = figure.add_subplot(223)
    plt.title("t = 40")
    plt.axis('off')
    ax3.imshow(pde_sp[1])

    ax4 = figure.add_subplot(224)
    plt.axis('off')
    ax4.imshow(utils.edge_detection(pde_sp[1]))

    plt.show()

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(221)
    plt.title("t = 80")
    plt.axis('off')
    ax1.imshow(pde_sp[2])

    ax2 = figure.add_subplot(222)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_sp[2]))

    ax3 = figure.add_subplot(223)
    plt.title("t = 120")
    plt.axis('off')
    ax3.imshow(pde_sp[3])

    ax4 = figure.add_subplot(224)
    plt.axis('off')
    ax4.imshow(utils.edge_detection(pde_sp[3]))

    plt.show()

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(121)
    plt.title("t = 160")
    plt.axis('off')
    ax1.imshow(pde_sp[4])

    ax2 = figure.add_subplot(122)
    plt.axis('off')
    ax2.imshow(utils.edge_detection(pde_sp[4]))

    plt.show()