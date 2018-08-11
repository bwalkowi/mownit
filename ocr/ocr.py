import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2

from skimage.io import imread
from skimage.filters import median, threshold_otsu
from skimage.transform import hough_line, hough_line_peaks, rotate
from skimage.morphology import disk
from skimage.feature import canny

from PIL.ImageFont import truetype
from PIL.Image import new
from PIL.ImageDraw import Draw


def denoise(image):
    """Reduce salt and peper noise.

    :param image: ndarray, 2D image
    :return: ndarray, 2D image with reduced salt and peper noise
    """
    return median(image, disk(2))


def integral_image(image):
    """Create integral image for given image.

    :param image: ndarray, 2D grayscale image
    :return:ndarray, 2D integral image
    """
    rows, cols = image.shape
    matrix = np.zeros((rows, cols, 2), dtype=np.uint64)

    matrix[0, 0, 0] = image[0, 0]
    matrix[0, 0, 1] = image[0, 0] ** 2
    for i in range(1, rows):
        matrix[i, 0, 0] = matrix[i-1, 0, 0] + image[i, 0]
        matrix[i, 0, 1] = matrix[i-1, 0, 1] + image[i, 0] ** 2
    for j in range(1, cols):
        matrix[0, j, 0] = matrix[0, j-1, 0] + image[0, j]
        matrix[0, j, 1] = matrix[0, j-1, 1] + image[0, j] ** 2

    for i in range(1, rows):
        for j in range(1, cols):
            matrix[i, j, 0] = matrix[i-1, j, 0] + matrix[i, j-1, 0] - \
                              matrix[i-1, j-1, 0] + image[i, j]
            matrix[i, j, 1] = matrix[i-1, j, 1] + matrix[i, j-1, 1] - \
                              matrix[i-1, j-1, 1] + image[i, j] ** 2
    return matrix


def threshold_local(image, k=0.35, window_size=(15, 15),
                    in_place=False, inverse=False):
    """Binarazie given image using local adaptive mean threshold.

    :param image: ndarray, 2D grayscale image
    :param k: initial threshold
    :param window_size: size of window
    :param in_place: whether modify existing image
    :param inverse: whether inverse colors
    :return: ndarray, 2D binarized image
    """
    int_img = integral_image(image)
#    min_intensity = np.amin(image)
    new_img = image if in_place else np.zeros(image.shape, dtype=np.uint8)
    min_val, max_val = (255, 0) if inverse else (0, 255)
    x, y = window_size
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            a = i + x // 2 if i + x // 2 < rows else rows - 1
            b = j + y // 2 if j + y // 2 < cols else cols - 1
            c = i - x // 2 if i - x // 2 > 0 else 0
            d = j - y // 2 if j - y // 2 > 0 else 0

            mean = int_img[a, b, 0] + int_img[c, d, 0] - int_img[a, d, 0] - int_img[c, b, 0]
            mean /= x * y

            dev = int_img[a, b, 1] + int_img[c, d, 1] - int_img[a, d, 1] - int_img[c, b, 1]
            dev /= x * y
            dev -= mean ** 2
            dev = np.sqrt(dev)

            # wolf
#            threshold = (1 - k) * mean
#            threshold += k * min_intensity
#            threshold += (k * dev) / (128 * (mean - min_intensity))
#            threshold = np.sqrt(threshold)

            threshold = k * (dev / 128 - 1) + 1
            threshold *= mean

            new_img[i, j] = min_val if image[i, j] < threshold else max_val
    return new_img


def threshold_global(image, in_place=False, inverse=False):
    """Binarazie given image using global otsu threshold.

    :param image: ndarray, 2D grayscale image
    :param in_place: whether modify existing image
    :param inverse: whether inverse colors
    :return: ndarray, 2D binarized image
    """
    threshold = threshold_otsu(image)
    new_img = image if in_place else np.zeros(image.shape, dtype=np.uint8)
    min_val, max_val = (255, 0) if inverse else (0, 255)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            new_img[i, j] = min_val if image[i, j] <= threshold else max_val
    return new_img


def deskew(image, edge=True):
    """Rotate image so that lines will be horizontally aligned.

    :param image: ndarray, 2D binarized image
    :return: ndarray, 2D deskewed image
    """
    rows, cols = image.shape
    image2 = canny(image, 5) if edge else image
    plt.imshow(image2, cmap='gray')
    plt.show()

    plt.imshow(image, cmap='gray')
    skew = 0
    _, angles, dists = hough_line_peaks(*hough_line(image2))
    for angle, dist in zip(angles, dists):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - cols * np.cos(angle)) / np.sin(angle)
        plt.plot((0, cols), (y0, y1), '-r')
        skew += 90 + np.rad2deg(angle) if angle < 0 else np.rad2deg(angle) - 90
    skew /= angles.shape[0] if angles.shape[0] != 0 else 1
    print(skew)
    plt.xlim([0, cols])
    plt.ylim([rows, 0])
    plt.show()
    return rotate(image, skew, resize=True, mode='constant',
                  cval=255, preserve_range=True).astype(np.uint8)


def crop_text(image):
    """Crop area of image with entire text within

    :param image: ndarray, 2D binarized image
    :return: ndarray, 2D binarized text image
    """
    rows, cols = image.shape
    a, b, c, d = rows, cols, 0, 0
    for i in range(0, rows):
        for j in range(0, cols):
            if image[i, j]:
                a = i if i < a else a
                b = j if j < b else b
                c = i if i > c else c
                d = j if j > d else d
    return image[a:c+1, b:d+1]


def match(origin, template):
    """Match origin image with template.

    :param origin: ndarray, 2D image
    :param template: ndarray, 2D image
    :return: max val of correlation
    """
    matrix = np.real(ifft2(np.multiply(fft2(origin), fft2(np.rot90(template, 2),
                                                          s=origin.shape))))
    return matrix.max()


def load_chars(font_name, size, chars):
    """Load characters from given font of given size.

    :param font_name: given font name
    :param size: given size
    :param chars: given set of characters
    :return: dict of tuples - character: (image, max_match_val)
    """
    font = truetype(font_name + ".ttf", size)
    symbols = {}
    for char in chars:
        width, height = font.getsize(char)
        img = new("1", (width, height), 0)
        drawing = Draw(img)
        drawing.text((0, 0), char, fill=255, font=font)
        img = img.convert("L")
        img = np.asarray(img, dtype=np.uint8)
        if char != ' ':
            img = crop_text(img)
        symbols[char] = (img, match(img, img))
    return symbols


def get_lines(image, fonts, chars):
    """Segment and return lines from image one by one.

    :param image: ndarray, 2D binarized, deskewed image
    :param fonts: names of fonts to use
    :param chars: characters in text
    :return: line
    """
    # horizontal projection profile
    hpp = [np.count_nonzero(row) for row in image]
    # plt.plot(np.arange(image.shape[0]), hpp)
    # plt.show()

    in_line = False
    start = 0
    lines = []
    line_height = 0
    for i, val in enumerate(hpp):
        if not in_line and val > 0:
            start = i
            in_line = True
        elif in_line and val == 0:
            lines.append((start, i))
            line_height += i - start
            in_line = False
        else:
            continue
    if hpp[-1] > 0:
        lines.append((start, len(hpp)))
        line_height += len(hpp) - start

    lines_num = len(lines)
    line_height //= lines_num
    yield [(size, {font: load_chars(font, size, chars) for font in fonts})
           for size in range(int(1.3*line_height), line_height, -1)]

    for i in range(lines_num - 1):
        yield image[lines[i][0]:lines[i][1], :], \
              "\n" * (1 + (lines[i + 1][0] - lines[i][1]) // line_height)

    yield image[lines[lines_num - 1][0]:lines[lines_num - 1][1], :], "\n"


def get_words(line, space_min_width):
    """Segment and return words from line image one by one.

    :param line: ndarray, 2D binarized, deskewed line image
    :param space_min_width: minimum width of space in font used
    :return: word
    """
    # vertical projection profile
    vpp = [np.count_nonzero(line[:, col]) for col in range(line.shape[1])]
    # plt.plot(np.arange(line.shape[1]), vpp)
    # plt.show()

    in_word = False
    start = 0
    end = 0
    words = []
    for i, val in enumerate(vpp):
        if not in_word and val > 0:
            in_word = True
            if i - end > space_min_width:
                words.append((start, end))
                start = i
        elif in_word and val == 0:
            end = i
            in_word = False
        else:
            continue
    if vpp[-1] == 0 and (start, end) not in words:
        words.append((start, end))
    elif vpp[-1] > 0 and len(vpp) - start > space_min_width:
        words.append((start, len(vpp)))

    for i in range(len(words) - 1):
        yield line[:, words[i][0]:words[i][1]], vpp[words[i][0]:words[i][1]], \
              " " * ((words[i + 1][0] - words[i][1]) // space_min_width)

    yield line[:, words[-1][0]:words[-1][1]], vpp[words[-1][0]:words[-1][1]], ""


def get_letters(word, vpp, font_height):
    """Segment and return letters from word image one by one.

    :param word: ndarray, 2D binarized, deskewed word image
    :param vpp: vertical projection profile of a word
    :param font_height: height of font used
    :return: letter image
    """
    cols = len(vpp)
    # plt.plot(np.arange(cols), vpp)
    # plt.show()
    width = int(0.8 * font_height)
    in_letter = False
    start = 0
    for i, val in enumerate(vpp):
        if not in_letter and val > 0:
            in_letter = True
            start = i
        elif in_letter and val == 0:
            in_letter = False
            if i - start <= width:
                yield word[:, start:i]
            else:
                letters = (i - start) // width + 1
                interval = (i - start) // letters
                for _ in range(letters):
                    yield word[:, start:start + interval]
                    start += interval
        else:
            continue
    if in_letter:
        yield word[:, start:cols]


def similar(origin, template):
    """Check similarity.

    :param origin: ndarray, 2D binarized image
    :param template: ndarray, 2D binarized image
    :return: whether images are similar
    """
    origin_h, origin_w = origin.shape
    template_h, template_w = template.shape
    width = origin_w if origin_w > template_w else template_w
    height = origin_h if origin_h > template_h else template_h
    tmp = np.zeros((height, width), dtype=np.uint8)
    tmp[0:origin_h, 0:origin_w] = origin
    tmp[0:template_h, 0:template_w] -= template
    if np.count_nonzero(tmp) > 0.1 * height * width:
        return False
    else:
        return True


def ocr(image, fonts_names, chars, threshold):
    """Turn image into doc.

    :param image: ndarray, 2D binarized text image
    :param fonts_names: fonts possibly used
    :param chars: characters possibly used
    :return: list of text lines
    """
    lines_gen = get_lines(image, fonts_names, chars)
    sizes_and_fonts = next(lines_gen)
    lines = [line for line in lines_gen]

    sign_histogram = {}
    doc = ""
    match_val = 0
    font_height = 0
    font_name = ""

    for font_size, fonts in sizes_and_fonts:
        for font in fonts:
            space_width = fonts[font][' '][0].shape[1]
            text = ""
            font_match_val = 0
            occurrences = {char: 0 for char in chars}
            for line, new_lines in lines:
                # plt.imshow(line, cmap='gray')
                # plt.show()
                for word, vpp, spaces in get_words(line, space_width):
                    # plt.imshow(word, cmap='gray')
                    # plt.show()
                    for letter in get_letters(word, vpp, font_size):
                        letter = crop_text(letter)
                        match_sign = ' '
                        score = 0
                        # plt.imshow(letter, cmap='gray')
                        # plt.show()
                        for char in chars:
                            # plt.imshow(fonts[font][char][0], cmap='gray')
                            # plt.show()
                            score = match(letter, fonts[font][char][0])
                            if score >= threshold * fonts[font][char][1]:
                                if similar(letter, fonts[font][char][0]):
                                    score = fonts[font][char][1]
                                    match_sign = char
                                    break

                        text += match_sign
                        if match_sign != ' ':
                            font_match_val += score
                        occurrences[match_sign] += 1
                        # plt.imshow(fonts[font][match_sign], cmap='gray')
                        # plt.show()
                    text += spaces
                text += new_lines
            if font_match_val > match_val:
                doc = text
                match_val = font_match_val
                font_name = font
                font_height = font_size
                sign_histogram = occurrences

    return doc, font_name, font_height, sign_histogram


def main():
    chars = "1839465072dbpqoeasmvxzgkhniljwyctfur?!,. "
    fonts = ["Arial", "Georgia", "FreeSans"]
    image = imread("text7.jpg", as_grey=True)
    threshold = 0.6

    # image = threshold_global(image)
    # image = denoise(image)
    # image = deskew(image)
    # image = denoise(image)
    tmp = threshold_global(image, inverse=True)
    tmp = crop_text(tmp)
    plt.imshow(tmp, cmap='gray')
    plt.show()

    text, font_name, font_height, sign_histogram = ocr(tmp, fonts, chars, threshold)
    print("Font: ", font_name, "\nFont height: ", font_height, sep='')
    print("Text:\n", text, sep='')

    if sign_histogram:
        x = np.arange(len(sign_histogram))
        plt.bar(x, sign_histogram.values(), width=0.5)
        plt.xticks(x, sign_histogram.keys())
        plt.xlim(0, len(x))
        plt.ylim(0, max(sign_histogram.values()) + 1)
        plt.show()


if __name__ == "__main__":
    main()
