"""CSV文件读取"""
import  csv
import numpy as np

def csvreader(fpath):
    """csv reader"""
    data = []
    target = []
    with open(fpath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            temp = []
            lists = list(row[0].split())
            for l in lists:
                temp.append(float(l))
                data.append(temp[0 : -1])
                target.append(temp[-1])
    return data, target

# def csvreader_split(fpath , split_rate):
#     data = []
#     target = []
#     with open(fpath,"r") as f:
#         reader = csv.reader(f)
#         for row in reader:
#             temp = []
#             lists = list(row[0].split())
#             for l in lists:
#                 temp.append(float(l))
#                 data.append(temp[0 : -1])
#                 target.append(temp[-1])
#     data = np.array(data)
#     target = np.array(target)
#     split_num = int(len(data) * split_rate)
#     train_data = data[ : split_num]
#     train_target = target[ : split_num]
#     test_data = data[split_num : ]
#     test_target = target[split_num : ]
#     return train_data , train_target , test_data , test_target

def csvreader_split(fpath, split_rate):
    """csv reader"""
    container = []
    data = []
    target = []
    with open(fpath, "r") as f:
        reader = csv.reader(f)
        for rows in reader:
            temp = []
            for num in rows[0].split():
                temp.append(float(num))
            container.append(temp)

    for rows in container:
        data.append(rows[0:-1])
        target.append(rows[-1])
    data = np.array(data)
    target = np.array(target)
    split_num = int(len(data) * split_rate)
    train_data = data[ : split_num]
    train_target = target[ : split_num]
    test_data = data[split_num : ]
    test_target = target[split_num : ]
    return train_data, train_target, test_data, test_target
