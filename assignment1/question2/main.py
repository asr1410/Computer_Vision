import cv2
import os
import numpy as np


def resize_image(image, new_row, new_col):
    org_row, org_col = image.shape[:2]
    resized_image = np.zeros((new_row, new_col, 3), dtype=np.uint8)
    scaled_row = org_row / new_row
    scaled_col = org_col / new_col
    for row in range(new_row):
        for col in range(new_col):
            resized_image[row, col] = image[
                int(scaled_row * row), int(scaled_col * col)
            ].astype(np.uint8)
    return resized_image


video_file_path = "/content/1 Minute Timer.mp4"
output_folder = "/content/Output"
mod_value = 500
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
capture = cv2.VideoCapture(video_file_path)
if not capture.isOpened():
    print("cannot open file")
    exit()
f_count = 0
while True:
    check, frame = capture.read()
    if not check:
        break
    f_count += 1
    if f_count % mod_value != 0:
        continue
    frame = resize_image(frame, 256, 256)
    f_name = os.path.join(output_folder, f"{f_count}.jpg")
    cv2.imwrite(f_name, frame)
    print(f"Saved {f_count} frame")
capture.release()
cv2.destroyAllWindows()
print(f"extracted frames saved to '{output_folder}'.")
