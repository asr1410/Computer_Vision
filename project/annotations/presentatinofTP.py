import csv
import os
import matplotlib.pyplot as plt
import cv2

file_path = 'annotations/annotations_test.csv'

# Specify the required entry
required_entry = 'test_0.jpg'
image_path = os.path.join("images", required_entry)
x1, y1, x2, y2, cls, img_width, img_height = [], [], [], [], [], [], []

with open(file_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        column_value = row[0]
        if column_value == required_entry:
            x1.append(int(row[1]))
            y1.append(int(row[2]))
            x2.append(int(row[3]))
            y2.append(int(row[4]))
            cls.append(str(row[5]))
            img_width.append(int(row[6]))
            img_height.append(int(row[7]))

x1p, y1p, x2p, y2p, con, hd, tp = [], [], [], [], [], [], []
file_path1 = 'working/detections_output_iou_0.5_with_mapping.csv'
with open(file_path1, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        column_value = row[0]
        if column_value == required_entry:
            x1p.append(int(row[1]))
            y1p.append(int(row[2]))
            x2p.append(int(row[3]))
            y2p.append(int(row[4]))
            con.append(float(row[5]))
            hd.append(float(row[6]))
            tp.append(float(row[7]))

image = plt.imread(image_path)

# Draw rectangles on the image based on xc, yc, h, and w values
fig, ax = plt.subplots()
ax.imshow(image)
count_gt = 0
for i in range(len(x1)):
    start_point = (int(x1[i]), int(y1[i]))
    end_point = (int(x2[i]), int(y2[i]))
    color = (0, 255, 0)
    thickness = 10
    im = cv2.rectangle(image, start_point, end_point, color, thickness)
    count_gt += 1
plt.imshow(im, label="Ground Truth")
# Draw rectangles on the image based on xc, yc, h, and w values
count_tp = 0
count_fp = 0
for i in range(len(x1p)):
    start_point = (int(x1p[i]), int(y1p[i]))
    end_point = (int(x2p[i]), int(y2p[i]))
    thickness = 10
    if tp[i] == 1:
        color = (255, 0, 0)
        im_tp = cv2.rectangle(image, start_point, end_point, color, thickness)
        count_tp += 1
    else:
        color = (0,0, 255)
        im_fp = cv2.rectangle(image, start_point, end_point, color, thickness)
        count_fp += 1
    
    

# Display the image using Matplotlib
plt.imshow(im_tp, label = "True Positive")
plt.imshow(im_fp, label = "False Positive")
plt.title("Ground Truth: " + str(count_gt) + ", True Positive: " + str(count_tp) + ", False Positive: " + str(count_fp))
plt.axis("off")
plt.savefig(f'working/ObjectDetected_{required_entry}', dpi = 1200)
plt.show()
