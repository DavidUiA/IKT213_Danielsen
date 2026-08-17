import numpy as np
import cv2

# Load a color image in grayscale
img = cv2.imread('iris-1.jpg')

def main():
    print_image_information(img)
    save_cam_info_to_file()

def show_iris_img():
    cv2.namedWindow('iris', cv2.WINDOW_NORMAL)  # spawner vinduet inn først
    cv2.imshow('iris', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def print_image_information(image):
    # channels is the 3rd value in shape, unless its grayscale,
    # then the 3rd value doesnt exist which means theres only 1 channel
    channels = img.shape[2] if len(img.shape) == 3 else 1
    print(f"height: {image.shape[0]}\n"
          f"width: {image.shape[1]}\n"
          f"channels: {channels}\n"
          f"size: {image.size}\n"
          f"data type: {image.dtype}\n")

def save_cam_info_to_file():
    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cam.get(cv2.CAP_PROP_FPS)

    cam_info = (f"fps: {camera_fps}\n"
                f"height: {frame_height}\n"
                f"width: {frame_width}\n")

    print(cam_info)

    with open("camera_outputs.txt", "w", encoding="utf-8") as f:
        f.write(cam_info)


    cam.release()
    cv2.destroyAllWindows()

main()