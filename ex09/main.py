import numpy as np
import PIL.ImageOps
from PIL import Image
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2


def main(image, pattern, text_img, treshold=0.95):
    img = Image.open(image).convert('L')
    template = Image.open(pattern).convert('L')
    if text_img:
        img = PIL.ImageOps.invert(img)
        template = PIL.ImageOps.invert(template)

    img_matrix = np.matrix(list(img.getdata()), float)
    img_matrix.shape = (img.size[1], img.size[0])

    template_matrix = np.matrix(list(template.getdata()), float)
    template_matrix.shape = (template.size[1], template.size[0])

    matrix = np.real(ifft2(np.multiply(fft2(img_matrix),
                                       fft2(np.rot90(template_matrix, 2),
                                            s=img_matrix.shape))))
    plt.imshow(matrix, cmap='gray')
    plt.show()

    matrix2 = np.absolute(matrix) > treshold*np.absolute(matrix).max()
    print("Number of occurrences of a pattern: %d" % np.count_nonzero(matrix2))
    plt.imshow(matrix2, cmap='gray')
    plt.show()


if __name__ == "__main__":
    # image, pattern, text, treshold = 'galia.png', 'galia_e.png', True, 0.95
    image, pattern, text, treshold = 'school.jpg', 'fish1.png', False, 0.7
    main(image, pattern, text, treshold)
