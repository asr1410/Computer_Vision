import pandas as pd

# Assuming your DataFrame is named 'df'
# Replace 'your_file.csv' with the actual file path

df_actual = pd.read_csv('./detections_output_iou_0.70_with_mapping.csv')
# Sort the DataFrame by the 'confidence' column in descending order
df = df_actual.sort_values(by='confidence', ascending=False)
# Initialize counters
accumulated_tp = 0
accumulated_fp = 0

df_gt = pd.read_csv('./annotations/annotations_test.csv', header=None, names=['filename', 'x_min_gt', 'y_min_gt', 'x_max_gt', 'y_max_gt', 'label', 'img_width', 'img_height'])

# Lists to store accumulated values
accumulated_tp_list = []
accumulated_fp_list = []

# Iterate through rows
for index, row in df.iterrows():
    if row['mapping'] == 1:
        # If mapping is 1, it's a True Positive
        accumulated_tp += 1
        accumulated_fp_list.append(accumulated_fp)
        accumulated_tp_list.append(accumulated_tp)
    else:
        # If mapping is 0, it's a False Positive
        accumulated_fp += 1
        accumulated_tp_list.append(accumulated_tp)
        accumulated_fp_list.append(accumulated_fp)

# Add new columns to the DataFrame
df['Accumulated_TP'] = accumulated_tp_list
df['Accumulated_FP'] = accumulated_fp_list

# Calculate Precision and Recall values
df['Precision'] = df['Accumulated_TP'] / (df['Accumulated_TP'] + df['Accumulated_FP'])
df['Recall'] = df['Accumulated_TP'] / len(df_gt)

# Save the updated DataFrame to a new CSV file
df.to_csv('./working/precision_recall_added_0.70_iou.csv', index=False)
