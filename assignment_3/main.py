import numpy as np
import cv2

img = cv2.imread('lambo.png')
shapesImg = cv2.imread('shapes.png', 0)
shapesImgRgb = cv2.imread('shapes.png')
template = cv2.imread('shapes_template.jpg', 0)


def main():
    cv2.imwrite("sobel.png", sobel_edge_detection(img))
    cv2.imwrite("edges.png", canny_edge_detection(img, 50, 50))
    cv2.imwrite("res.png", template_match(shapesImg, template))
    cv2.imwrite("resized_up.png", resize(img, 2, "up"))
    cv2.imwrite("resized_down.png", resize(img, 2, "down"))

    #cv2.imshow("", edges)
    #cv2.waitKey(0)

def sobel_edge_detection(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.magnitude(sobelx, sobely)  # Euclidean gradient magnitude
    normalized = cv2.normalize(sobelxy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized

def canny_edge_detection(image, threshold_1, threshold_2):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(img_blur, threshold_1, threshold_2)
    return edges

def template_match(image, template):
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(res >= threshold)

    h, w = template.shape[:2]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(shapesImgRgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    return shapesImgRgb

def resize(image, scale_factor: int, up_or_down: str):
    for _ in range(scale_factor):
        rows, cols, _channels = map(int, image.shape)
        if up_or_down == "up":
            image = cv2.pyrUp(image, dstsize=(2 * cols, 2 * rows))
        elif up_or_down == "down":
            image = cv2.pyrDown(image, dstsize=(cols // 2, rows // 2))
        else:
            return None
    return image


main()
