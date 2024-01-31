import numpy as np
import pandas as pd

# # Load ground truth and predicted box coordinates from CSV files
ground_truth_df = pd.read_csv('./annotations/annotations_test.csv', header=None, names=['filename', 'x_min_gt', 'y_min_gt', 'x_max_gt', 'y_max_gt', 'label', 'img_width', 'img_height'])
predicted_df = pd.read_csv('./detections_output_iou_0.80_with_mapping.csv', header=None, names=['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'confidance', 'hard'])

# extract first column from ground truth
distinct_images = ground_truth_df['filename'].unique()
# print(len(distinct_images))
MAE = 0
MSE = 0

for images in distinct_images:
    images_ground_truth = ground_truth_df[ground_truth_df['filename'] == images]
    images_predicted = predicted_df[predicted_df['filename'] == images]
    
    MSE = MSE + (len(images_ground_truth) - len(images_predicted))**2
    MAE = MAE + abs(len(images_ground_truth) - len(images_predicted))
    

MSE = MSE/len(distinct_images)
MAE = MAE/len(distinct_images)
RMSE = np.sqrt(MSE)

with open('./working/RMSE_MAE1_0.80.txt', 'w') as f:
    f.write('RMSE: ' + str(RMSE) + '\n')
    f.write('MAE: ' + str(MAE) + '\n')