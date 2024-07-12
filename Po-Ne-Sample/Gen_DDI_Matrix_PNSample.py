
import pandas as pd
import csv
import numpy as np

def Interections():#根据两个原始数据得到841*841的关系矩阵
    # Generate drug_drug interactions matrix
    M = []
    N = []
    dic = {}

    with open('D:/StudyFile/Acasci/DNGR+ANN+DNN/data/dataset/Node_Codes.csv', "rt", encoding='utf-8')as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            M.append(i[0])
            N.append(i[1])

        for i in range(len(M)):
            dic[M[i]] = N[i]

    D = []
    I = []

    with open('D:/StudyFile/Acasci/DNGR+ANN+DNN/data/dataset/Drug_Information.csv', "rt", encoding='utf-8')as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            D.append(i[0])#第一列
            I.append(i[1])#第二列

    DDI = np.zeros((len(D), len(D)), int)

    for i in range(len(D)):
        for j in I[i].split('|'):
            if not j.strip() == '' and j in M:
                DDI[int(dic[D[i]]) - 1][int(dic[j]) - 1] = 1

    # 将 DDI 转换为 pandas DataFrame
    df = pd.DataFrame(DDI)
    # 指定要保存的 CSV 文件名
    csv_file = "11111excle_DDI_matrix.csv"
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv(csv_file, index=False, header=False)
    print(f"CSV 文件 {csv_file} 已成功保存。")
    return DDI


drug_drug_matrix = Interections()
link_number = 0
link_position = []
nonLinksPosition = []
for i in range(0, len(drug_drug_matrix)):
    for j in range(i + 1, len(drug_drug_matrix)):  # 循环矩阵的上三角
        if drug_drug_matrix[i, j] == 1:
            link_number = link_number + 1
            link_position.append([i, j])
        elif drug_drug_matrix[i, j] == 0 and np.sum(drug_drug_matrix[i, :], axis=0) > 0 and np.sum(
                drug_drug_matrix[:, j], axis=0) > 0:
            nonLinksPosition.append([i, j])
link_position = np.array(link_position)  # 把列表的每个元素[0,31]......加入数组中去，也就是每个关系1的位置

# 指定要保存的CSV文件的文件名
file_name1 = '111111111Positive_Sample.csv'
# 使用CSV模块创建一个CSV文件并写入数据
with open(file_name1, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 遍历数组，并将每个子数组写入CSV文件中的一行
    for row in link_position:
        writer.writerow(row)
print(f'CSV文件正样本 "{file_name1}" 已成功创建!')

# 指定要保存的CSV文件的文件名
file_name2 = '1111111Negative_Sample_All.csv'
# 使用CSV模块创建一个CSV文件并写入数据
with open(file_name2, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 遍历数组，并将每个子数组写入CSV文件中的一行
    for row in nonLinksPosition:
        writer.writerow(row)
print(f'CSV文件负样本 "{file_name2}" 已成功创建!')



# 读取all.csv文件
all_data = pd.read_csv('Negative_Sample_All.csv')

# 随机挑选82620行数据
random_data = all_data.sample(n=82620, random_state=42)  # 使用random_state以确保结果可复现

# 将选定的数据存储到111111.csv文件中
random_data.to_csv('Negative_Sample82620.csv', index=False)

# 读取 data1.csv 和 data2.csv 文件
data1 = pd.read_csv('Negative_Sample82620.csv')
data2 = pd.read_csv('Positive_Sample.csv')

# 合并两个数据框
merged_data = pd.concat([data1, data2], axis=0)

# 将合并后的数据保存为 data.csv
merged_data.to_csv('ourdata_dd.csv', index=False)


# 读取CSV文件
merged_data = pd.read_csv('ourdata_dd.csv')

# 保存为文本文件（.txt），使用制表符分隔数据
merged_data.to_csv('ourdata_dd.txt', sep='\t', index=False)