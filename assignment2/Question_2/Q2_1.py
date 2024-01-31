import cv2
import numpy as np

def find_keypoints_and_descriptors(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def find_good_matches(descriptors1, descriptors2):
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict())
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    return good_matches

def find_corresponding_points(image1, image2):
    keypoints1, descriptors1 = find_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = find_keypoints_and_descriptors(image2)
    
    good_matches = find_good_matches(descriptors1, descriptors2)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return points1, points2

def compute_essential_matrix(points1, points2):
    fundamental_matrix, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_LMEDS)
    return fundamental_matrix

def rectify_images(image1, image2, points1, points2):
    fundamental_matrix = compute_essential_matrix(points1, points2)
    _, homography1, homography2 = cv2.stereoRectifyUncalibrated(points1, points2, fundamental_matrix, image1.shape[:2])

    rectified_image1 = cv2.warpPerspective(image1, homography1, image1.shape[:2][::-1])
    rectified_image2 = cv2.warpPerspective(image2, homography2, image2.shape[:2][::-1])

    return rectified_image1, rectified_image2

def main(image_path1, image_path2):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    points1, points2 = find_corresponding_points(image1, image2)
    rectified_image1, rectified_image2 = rectify_images(image1, image2, points1, points2)

    cv2.imshow('Original Image 1', image1)
    cv2.imshow('Original Image 2', image2)
    cv2.imshow('Rectified Image 1', rectified_image1)
    cv2.imshow('Rectified Image 2', rectified_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    image_path1 = './sample_left.png'
    image_path2 = './sample_right.png'

    main(image_path1, image_path2)