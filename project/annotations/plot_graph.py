import matplotlib.pyplot as plt
import pandas as pd

# Load the updated CSV file
df = pd.read_csv('./working/precision_recall_added_0.80_iou.csv')

# Plot the Precision-Recall curve
plt.plot(df['Recall'], df['Precision'])
plt.xlabel('Recall')
plt.ylabel('Precision')

#Average Precision using All Points Interpolation
# Initialize variables
previous_recall_value = 0
previous_precision_value = 1
area_under_curve = 0

# Iterate through rows
for index, row in df.iterrows():
    if row['Recall'] != previous_recall_value:
        # Calculate area of trapezoid
        area = (row['Recall'] - previous_recall_value) * ((row['Precision'] + previous_precision_value) / 2)
        area_under_curve += area
        # Update previous values
        previous_recall_value = row['Recall']
        previous_precision_value = row['Precision']

print('Average Precision using All Points Interpolation: ' + str(area_under_curve))
plt.title('Precision-Recall Curve with Average Precision of {:.3f}'.format(area_under_curve))
plt.savefig('./working/Precision-Recall 0.80 iou Curve.jpg', dpi=1200)
plt.show()