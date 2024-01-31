'''
Miss Rate is a ratio of false negetive to the total number of positive objects in the ground truth i.e. FN/(TP+FN).

Miss Rate = FN / (TP+FN) = #GT - TP / #GT
'''
# False positive per image is a ratio of false positive to the total number of tested images.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load ground truth and predicted box coordinates from CSV files
predicted_df = pd.read_csv('./working/precision_recall_added_0.80_iou.csv')
ground_truth_df = pd.read_csv('./annotations/annotations_test.csv', header=None, names=['filename', 'x_min_gt', 'y_min_gt', 'x_max_gt', 'y_max_gt', 'label', 'img_width', 'img_height'])

# extract first column from ground truth
distinct_images = predicted_df['filename'].unique()

predicted_df['MissRate'] = (len(ground_truth_df) - predicted_df['Accumulated_TP']) / len(ground_truth_df)
predicted_df['FPPI'] = predicted_df['Accumulated_FP'] / len(distinct_images)


# Plot the log-scale graph
plt.plot(np.log10(predicted_df['FPPI']), (predicted_df['MissRate']))
plt.title('MR-FPPI Plot')
plt.xlabel('False Positive Per Image')
plt.ylabel('Miss Rate')
plt.xticks(np.arange(0, 3), [f'$10^{i}$' for i in range(0, 3)])
# plt.yticks(np.arange(.4, 1.2, .2), [f'${i}x10^{-1}$' for i in [4, 6, 8, 10]])
plt.xlim(-1, 2)
plt.savefig('./working/MR-FPPI iou 80 Plot.jpg', dpi=1200)
plt.show()