import os
import numpy as np

src_dir = 'trained_weights'
files = os.listdir(src_dir)
loss_list, acc_list = [], []

result_file = 'model_files.txt'

for file_name in files:
    name, _ = os.path.splitext(file_name)
    name_parts = name.split('-')
    loss_list.append(float(name_parts[-2]))
    acc_list.append(float(name_parts[-1]))

loss_list = np.array(loss_list)
acc_list = np.array(acc_list)

sums = 1 + loss_list - acc_list

model_files = []

for i in range(len(files)):
    model_files.append((sums[i], files[i]))

limit = 5
model_files.sort()
print(model_files[:limit])

with open(result_file, 'w') as f:
    for i in range(limit):
        f.write(model_files[i][-1] + '\n')