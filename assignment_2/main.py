import numpy as np
import cv2

img = cv2.imread('../iris.jpg')


def main():
    cv2.imwrite("padding.png", padding(img, 100))
    cv2.imwrite("resized.png", resize(img, 200, 200))
    cv2.imwrite("cropped.png", crop(img, 200, img.shape[1]-130, 200, img.shape[0]-130))
    cv2.imwrite("greyscale.png", greyscale(img))
    cv2.imwrite("hsv.png", hsv(img))
    cv2.imwrite("hue_shifted.png", hue_shifted(img, np.zeros_like(img), 50))
    cv2.imwrite("smoothing.png", smoothing(img))
    cv2.imwrite("rotated_90.png", rotation(img, 90))
    cv2.imwrite("rotated_180.png", rotation(img, 180))


def padding(image, border_width):
    return cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)


def crop(image, x0, x1, y0, y1):
    return image[y0:y1, x0:x1]


def resize(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def copy(image, empty_picture_array):
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            empty_picture_array[x][y] = image[x][y]


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def hue_shifted(image, empty_picture_array, hue):
    hsv_img = hsv(img)
    for y in hsv_img:
        for x in y:
            x[0] += hue
            x[0] %= 180  # 180 instead since 360 doesnt fit in a byte
            x[2] = min(int(x[2]) + 50, 255)
    copy(hsv_img, empty_picture_array)
    return empty_picture_array


def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)


def rotation(image, rotation_angle):
    angle = "error, wrong angle"
    if rotation_angle == 90: angle = cv2.ROTATE_90_CLOCKWISE
    if rotation_angle == 180: angle = cv2.ROTATE_180

    return cv2.rotate(image, angle)


main()
