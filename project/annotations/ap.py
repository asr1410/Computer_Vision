import pandas as pd

avg_recall_list  = []

iou_values = [50, 55, 60, 65, 75, 85, 90, 95]

for iou in iou_values:
    # read a csv flie with header
    df = pd.read_csv(f'./working/precision_recall_added_0.{iou}_iou.csv', header=0)

    # taken filename and confidence and mapping
    df = df.iloc[:, [0, 5, 7]]

    # take distinct filname from the dataframe and store in a list
    filenames = df['filename'].unique().tolist()

    recall_list = []

    # for each filename, take first 300 rows to calculate recall
    for filename in filenames:
        df_temp = df[df['filename'] == filename]
        df_temp = df_temp.sort_values(by=['confidence'], ascending=False)
        if len(df_temp) > 300:
            df_temp = df_temp.iloc[:300, :]
        recall_list.append(df_temp['mapping'].mean())

    # calculate average recall
    avg_recall = sum(recall_list) / len(recall_list)
    print(avg_recall)
    avg_recall_list.append(avg_recall)

print(sum(avg_recall_list) / len(avg_recall_list))