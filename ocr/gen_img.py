from PIL.ImageFont import truetype
from PIL.Image import new
from PIL.ImageDraw import Draw
from skimage.io import imsave
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate


def gen_img(font_name, font_size, text, img_size):
    font = truetype(font_name + ".ttf", font_size)
    print(font.getsize("By"))
    width, height = img_size

    img = new("1", (width, height), 255)
    drawing = Draw(img)
    drawing.text((10, 10), text, fill=0, font=font)
    img = img.convert("L")
    img = np.asarray(img, dtype=np.uint8)
    return img


def main():
    font_name = 'Georgia'
    # font_name = 'Ubuntu'
    font_size = 12
    img_size = (400, 80)
    text = "lorem ipsum et nihil humanum\name alienum esse puto dolor sit amet,\nconsecteur adipisicing elit, sed\n\ndo eubismod."
    img = gen_img(font_name, font_size, text, img_size)
    # img = rotate(img, 10, resize=True, mode='constant', cval=255, preserve_range=True).astype(np.uint8)
    plt.imshow(img, cmap='gray')
    plt.show()
    imsave("img3.png", img)


if __name__ == "__main__":
    main()
