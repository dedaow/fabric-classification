import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    root_dir = r'C:\Users\lenovo\Desktop\goodjob\current_posess1\csv_files'
    for filename in os.listdir(root_dir):
        df = pd.read_csv(os.path.join(root_dir, filename))
        df.fillna(0, inplace=True)
        data = np.squeeze(df.values)
        data_cleaned = np.zeros(data.shape[0], dtype=float)
        if (data.shape[0] < 300):
            os.remove(os.path.join(root_dir, filename))
            print(f'Delete {filename} successfully.')
        # if (data.ndim > 1):
        #     for i in range(data.shape[0]):
        #         if (data[i, 0] != 0):
        #             data_cleaned[i] = data[i, 0]
        #         elif (data[i, 1] != 0):
        #             data_cleaned[i] = data[i, 1]
        #         else:
        #             data_cleaned[i] = 0
        #     if (data_cleaned.std() == 0):
        #         print(filename)
        # if (df.shape[1] > 1):
        #     print(filename)
        #     for line in data:
        #         print(line)
        #     mean = data.mean()
        #     std = data.std()
        for filename in os.listdir(root_dir):
            if names[6] in filename:
                label = 6
                self.labels.append(label)
            elif names[11] in filename:
                label = 11
                self.labels.append(label)
            elif names[12] in filename:
                label = 12
                self.labels.append(label)
            elif names[15] in filename:
                label = 15
                self.labels.append(label)
            elif names[16] in filename:
                label = 16
                self.labels.append(label)