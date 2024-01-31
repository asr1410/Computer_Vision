import cv2
import numpy as np

def detect_harris_corners(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)
    corner_response = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    corner_response = cv2.dilate(corner_response, None)
    image[corner_response > 0.01 * corner_response.max()] = [0, 0, 255]  # Mark corners in red

    return image

def main_harris_corner_detection(image_path):
    image = cv2.imread(image_path)
    corners_image = detect_harris_corners(image)

    cv2.imshow('Harris Corner Detection', corners_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path = './sample_left.png'
    main_harris_corner_detection(image_path)