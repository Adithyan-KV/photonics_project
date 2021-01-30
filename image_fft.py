import skimage.io as io
import skimage.color as col
import scipy.fftpack as fft
import numpy as np
import matplotlib.pyplot as plt

def main():
    image = io.imread('art.jpg')
    grayscale_image = col.rgb2gray(image)
    _,spectrum = get_fourier_spectrum(grayscale_image)
    plot_side_by_side(image,spectrum,'2D-DFT','Original image','DFT spectrum (centered)')

    sine_image = generate_sinusoidal_image(300,300)
    _, spectrum_sin = get_fourier_spectrum(sine_image)
    plot_side_by_side(sine_image, spectrum_sin, '2D-DFT', 'Sinusoidal Image', 'DFT spectrum(centered)')    

def plot_side_by_side(image_1,image_2,main_title,title_1,title_2):
    # assuming equal size images. 
    label_limit_x = int(image_1.shape[1]/2)
    label_limit_y = int(image_1.shape[0]/2)
    fig, plots = plt.subplots(1,2)
    fig.suptitle(main_title)
    plots[0].imshow(image_1, extent=[-label_limit_x,label_limit_x,-label_limit_y,label_limit_y])
    plots[0].set_title(title_1)
    plots[1].imshow(image_2, cmap='gray',extent=[-label_limit_x,label_limit_x,-label_limit_y,label_limit_y])
    plots[1].set_title(title_2)
    plt.show()

def get_fourier_spectrum(image):
    fft_image = fft.fft2(image)
    fft_shifted = fft.fftshift(fft_image)
    fft_spectrum = np.log(1+np.abs(fft_shifted))
    return fft_image, fft_spectrum

def generate_sinusoidal_image(M, N):
    sin_image = np.zeros((M, N))
    rows, columns = sin_image.shape
    u = 100
    v = 200
    for m in range(rows):
        for n in range(columns):
            sin_image[m, n] = np.sin(2 * np.pi * (u * m / M + v * n / N))
    return sin_image * 255

if __name__ == '__main__':
    main()