import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import Utilities as utils

def coeff(I, k):

    return 1 / (1 + ((I/k) ** 2))

def diffuse(img, log_f, iter, k, lmb = 0.01):

    image = np.array(Image.open(img).convert('L')) / 255
    new_image = np.zeros(image.shape, dtype=image.dtype)

    result = [image]

    for t in range(iter):
        I_Up = image[:-2, 1:-1] - image[1:-1, 1:-1]
        I_Down = image[2:, 1:-1] - image[1:-1, 1:-1]
        I_Right = image[1:-1, 2:] - image[1:-1, 1:-1]
        I_Left = image[1:-1, :-2] - image[1:-1, 1:-1]

        new_image[1:-1, 1:-1] = image[1:-1, 1:-1] + lmb * (
            coeff(I_Up, k) * I_Up +
            coeff(I_Down, k) * I_Down +
            coeff(I_Right, k) * I_Right +
            coeff(I_Left, k) * I_Left
        )

        image = new_image

        if (t+1) % log_f == 0:
            result.append(image.copy())

    return result

def run_PM(og_img, img_gauss, img_sp):

    iterations = 80
    k_gauss = 0.1
    num_col = 5

    print('\t Running Perona-Malik for Gaussian Image \n')
    pde_gauss = diffuse(img_gauss, log_f= iterations / (num_col - 1), iter=iterations, k=k_gauss, lmb=0.1)

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

    print('\t Running Perona-Malik for Salt and Pepper Image \n')

    k_sp = 0.2
    iterations_sp = 160
    num_col = 5

    pde_sp = diffuse(img_sp, log_f= iterations_sp / (num_col - 1), iter=iterations_sp, k=k_sp, lmb=0.1)

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

    og_image = np.array(Image.open(og_img).convert('L')) / 255

    gauss_PSNR = []
    sp_PSNR = []

    for i in pde_gauss:
        gauss_PSNR.append(utils.PSNR(og_image, i))
    
    for i in pde_sp:
        sp_PSNR.append(utils.PSNR(og_image, i))
    
    figure = plt.figure()
    x = np.arange(0, iterations + iterations / (num_col - 1), iterations / (num_col - 1))
    x1 = np.arange(0, iterations_sp + iterations_sp / (num_col - 1), iterations_sp / (num_col - 1))

    ax1 = figure.add_subplot(2, 1, 1)
    ax1.plot(x, gauss_PSNR, '.-')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title('PSNR for Gaussian Image with k = 0.1')

    ax2 = figure.add_subplot(2, 1, 2)
    ax2.plot(x1, sp_PSNR, '.-')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title('PSNR for Salt and Pepper Image with k = 0.1')

    figure.tight_layout()
    plt.show()

def compare_k(og_img, img_gauss, img_sp):

    iterations_gaussian = 80
    iterations_sp = 160
    k = np.arange(0.01, 0.61, 0.01)
    lamb = 0.1

    result_image_gaussian = []
    result_psnr_gaussian = []

    result_image_sp = []
    result_psnr_sp = []

    original_image = np.array(Image.open(og_img).convert('L')) / 255

    for each_k in k:
        log = diffuse(img_gauss, log_f=iterations_gaussian,
                                    iter=iterations_gaussian,
                                    k=each_k, lmb=lamb)
        result_image_gaussian.append(log[-1])
        result_psnr_gaussian.append(utils.PSNR(log[-1], original_image))

        log = diffuse(img_sp, log_f=iterations_sp,
                                    iter=iterations_sp,
                                    k=each_k, lmb=lamb)
        result_image_sp.append(log[-1])
        result_psnr_sp.append(utils.PSNR(log[-1], original_image))

    print("\tPSNR for gaussian:", result_psnr_gaussian)
    print("\tPSNR for salt & pepper", result_psnr_sp)

    figure = plt.figure()
    plt.axis('off')
    plt.gray()

    ax1 = figure.add_subplot(2, 1, 1)
    ax1.plot(k, result_psnr_gaussian, ".-")
    plt.xlabel("k")
    plt.ylabel("PSNR/dB")

    ax2 = figure.add_subplot(2, 1, 2)
    ax2.plot(k, result_psnr_sp, ".-")
    plt.xlabel("k")
    plt.ylabel("PSNR/dB")

    figure.tight_layout()
    plt.show()

    print("")