# coding=utf-8

# all data path
data_root_path = "./data/"
data_raw_path = data_root_path + "Training_4541/"
numbering_path = data_raw_path + "numbering.txt"
tensors_BB_path = data_raw_path + "tensors_BB.txt"
tensors_F_path = data_raw_path + "tensors_F.txt"
tensors_HH_path = data_raw_path + "tensors_HH.txt"
tensors_P_path = data_raw_path + "tensors_P.txt"

input_data_path = "./data/input_data.csv"

def preprocess():
    """
    merge different data txt to one file.csv
    
    :return: 
    """
    data_save = []
    with open(numbering_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            # print(line.strip())
            # line_split = line.split('')
            print(line.strip())
            data_save.append(line.strip())
            # input()
    with open(tensors_BB_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            # print(line.strip())
            line_split = ','.join(line.strip().split())
            print(line_split)
            data_save[i] += ',' + line_split
            i += 1
            # input()
    with open(tensors_F_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            # print(line.strip())
            line_split = ','.join(line.strip().split())
            print(line_split)
            data_save[i] += ',' + line_split
            i += 1
            # input()
    with open(tensors_HH_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            # print(line.strip())
            line_split = ','.join(line.strip().split())
            print(line_split)
            data_save[i] += ',' + line_split
            i += 1
            # input()
    with open(tensors_P_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            # print(line.strip())
            line_split = ','.join(line.strip().split())
            print(line_split)
            data_save[i] += ',' + line_split
            i += 1
            # input()
    with open(input_data_path, "w", encoding="utf-8") as input_data_file:
        for line in data_save:
            input_data_file.write(line + "\n")
            pass

preprocess()