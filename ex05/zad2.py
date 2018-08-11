import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    img = Image.open("shiz.png")
    img_gray = img.convert('LA')
    plt.figure(figsize=(9, 6))
    plt.imshow(img_gray)
    plt.show()

    byte_matrix = np.array(list(img_gray.getdata(band=0)), float)
    byte_matrix.shape = (img_gray.size[1], img_gray.size[0])
    byte_matrix = np.matrix(byte_matrix)
    plt.figure(figsize=(9, 6))
    plt.imshow(byte_matrix, cmap='gray')
    plt.show()

    u, s, v = np.linalg.svd(byte_matrix)

    approx = np.matrix(u[:, :1]) * np.diag(s[:1]) * np.matrix(v[:1, :])
    plt.imshow(approx, cmap='gray')
    plt.show()

    for i in range(2, 4):
        approx = np.matrix(u[:, :i]) * np.diag(s[:i]) * np.matrix(v[:i, :])
        plt.imshow(approx, cmap='gray')
        title = "n = %s" % i
        plt.title(title)
        plt.show()

    for i in range(5, 71, 10):
        approx = np.matrix(u[:, :i]) * np.diag(s[:i]) * np.matrix(v[:i, :])
        plt.imshow(approx, cmap='gray')
        title = "n = %s" % i
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    main()
