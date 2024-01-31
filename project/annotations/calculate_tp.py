import pandas as pd
import matplotlib.pyplot as plt

def calculate_iou(box1, box2):
        x1, y1, w1, h1 = box1[0], box1[1], box1[2] - box1[0], box1[3] - box1[1]
        x2, y2, w2, h2 = box2[0], box2[1], box2[2] - box2[0], box2[3] - box2[1]

        # Calculate intersection area
        intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

        # Calculate union area
        union_area = (w1 * h1) + (w2 * h2) - intersection_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

def calculate_tp(ground_truth_df, predicted_df):
    # extract first column from ground truth
    distinct_images = ground_truth_df['filename'].unique()


    # Define IoU threshold
    iou_threshold = 0.75

    positive_mapping = []

    for images in distinct_images:
        images_ground_truth = ground_truth_df[ground_truth_df['filename'] == images]
        images_predicted = predicted_df[predicted_df['filename'] == images]
        
        for pr in range(len(images_predicted)):
            iou_score = 0
            
            for gd in range(len(images_ground_truth)):
                iou_score = max(iou_score, calculate_iou(images_ground_truth.iloc[gd, 1:5], images_predicted.iloc[pr, 1:5]))
            
            if iou_score > iou_threshold:
                positive_mapping.append(1)
            else:
                positive_mapping.append(0)

    # Insert a new column 'positive_mapping' in the predicted_df DataFrame
    predicted_df['mapping'] = positive_mapping

    # Save the updated DataFrame to a new CSV file
    predicted_df.to_csv('working/detections_output_iou_0.75_with_mapping.csv', index=False)


# Load ground truth and predicted box coordinates from CSV files
ground_truth_df = pd.read_csv('annotations/annotations_test.csv', header=None,
                                names=['filename', 'x_min_gt', 'y_min_gt', 'x_max_gt', 'y_max_gt', 'label', 'img_width', 'img_height'])
predicted_df = pd.read_csv('working/detections_output_iou_0.5_Sun_Dec_24_06_03_25_2023.csv', header=None,
                            names=['filename', 'x_min_pred', 'y_min_pred', 'x_max_pred', 'y_max_pred', 'confidence', 'hard_score'])

calculate_tp(ground_truth_df, predicted_df)