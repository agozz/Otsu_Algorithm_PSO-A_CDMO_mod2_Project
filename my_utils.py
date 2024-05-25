from my_PSO import Space
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time


def basic_otsu_algorithm(image):

    sig_quadroW = []

    histogram = np.zeros(256, dtype=np.int32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            value = image[i, j]

            histogram[value] += 1

    for t in range(0, 256):

        N1 = sum(histogram[0:t - 1]) + 0.1
        N2 = sum(histogram[t:255]) + 0.1

        N = image.shape[0] * image.shape[1]

        q1 = (N1 / N) + 0.1
        q2 = (1 - q1) + 0.1

        mu1 = 0
        mu2 = 0

        sig_quadro1 = 0
        sig_quadro2 = 0

        for i in range(0, t):
            mu1 += i * (histogram[i] / N1)

        for j in range(t, 256):
            mu2 += j * (histogram[j] / N2)

        for i in range(0, t):
            sig_quadro1 += ((i - mu1) ** 2) * ((histogram[i] / N) * (1 / q1))

        for j in range(t, 256):
            sig_quadro2 += ((j - mu2) ** 2) * ((histogram[j] / N) * (1 / q2))

        sig_quadroW.append(q1 * sig_quadro1 + q2 * sig_quadro2)

    return sig_quadroW.index(min(sig_quadroW))


def otsu_comparison(image, number_of_particles, stamp):

    start1 = time.time()
    space = Space(image=image, n_particles=number_of_particles)
    threshold_PSO = np.round(space.search_PSO())
    end1 = time.time()

    time_PSO = end1 - start1

    start2 = time.time()
    threshold_basicOtsu = basic_otsu_algorithm(image)
    end2 = time.time()

    time_basicOtsu = end2 - start2

    print("     This is the threshold obtained with the Basic Otsu's algorithm in", round(time_basicOtsu, 3), "s:",
          threshold_basicOtsu)
    print("     This is the threshold obtained with the PSO-Otsu's algorithm in", round(time_PSO, 3), "s: ",
          threshold_PSO)

    if stamp:

        print_images(image, threshold_PSO, threshold_basicOtsu)

    difference_times = time_PSO / time_basicOtsu

    return difference_times


def basicOtsu_vs_psoOtsu(list_of_images, number_of_particles=10, stamp=False):

    list_of_times = []

    for index, image in enumerate(list_of_images):
        print("Image", index)

        rel_between_times = otsu_comparison(image, number_of_particles=number_of_particles, stamp=stamp)
        list_of_times.append(rel_between_times)

    return round(sum(list_of_times) / len(list_of_times), 3)


def print_images(image, threshold_PSO, threshold_basicOtsu, x=500, y=55):

    image_copy1 = image.copy()
    image_copy2 = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            if image[i, j] > threshold_PSO:
                image_copy1[i, j] = 0

            else:
                image_copy1[i, j] = 255

            if image[i, j] > threshold_basicOtsu:
                image_copy2[i, j] = 0
            else:
                image_copy2[i, j] = 255

    fig = plt.figure(figsize=(10, 5))

    backend = matplotlib.get_backend()

    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))

    elif backend == 'WXAgg':
            fig.canvas.manager.window.SetPosition((x, y))

    else:
        fig.canvas.manager.window.move(x, y)

    fig.add_subplot(1, 3, 1)
    plt.title('Original image')
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(1, 3, 2)
    plt.title('Basic Otsu')
    plt.imshow(image_copy2, cmap='gray', vmin=0, vmax=255)

    fig.add_subplot(1, 3, 3)
    plt.title('PSO-Otsu')
    plt.imshow(image_copy1, cmap='gray', vmin=0, vmax=255)

    plt.show()
