import os
import scipy.io
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import csv
import random
import sys
sys.path.append("..")
import graph, dngr
import keras
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import datetime
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


def loadmodel(path):
    # Load the files of representation vectors generated before
    Emb = pd.read_csv(path, header=None)
    features = list(Emb.columns)
    Emb = np.array(Emb[features])
    return Emb


g = graph.Graph()  # 创建一个图，将其赋值为变量g
g.read_edgelist('D:\StudyFile\Ours\data\dataset\ourdata_dd.txt')  # 文档里面有165240对DDI，读取它做为图的边，并将其中的边添加到图g中。

#g.read_edgelist('D:\StudyFile\Acasci\DNGR+ANN+DNN\data\dataset\drug_drug.txt')  # 文档里面有165240对DDI，读取它做为图的边，并将其中的边添加到图g中。
#print(g.G.number_of_edges())  # 打印图中边的数量
# Obtain representation vectors by DNGR
print("Test Begin")
# model = sdne.SDNE(g, [1000, 64],)
model = dngr.DNGR(g, 4, 64, XY=None)
print("Test End")
data = pd.DataFrame(model.vectors).T
data.to_csv('dngr_16w_embedding_dd64.csv', header=None)


model_s = loadmodel('D:\StudyFile\Ours\data\embeddings\Dimension\dngr_ds1664.csv')
model_t = loadmodel('D:\StudyFile\Ours\data\embeddings\Dimension\dngr_dt1664.csv')
model_e = loadmodel('D:\StudyFile\Ours\data\embeddings\Dimension\dngr_de1664.csv')
model_p = loadmodel('D:\StudyFile\Ours\data\embeddings\Dimension\dngr_dp1664.csv')

#model = loadmodel('D:\StudyFile\Acasci\DNGR+ANN+DNN\src\FeatureLink\dngr_16w_embedding_dd128.csv')

I1 = []
with open('dngr_16w_embedding_dd64.csv', "rt", encoding='utf-8') as csvfile1:
    reader = csv.reader(csvfile1)
    for i in reader:
        I1.append(i[0])  # 第一个元素加入到列表中，也就是药物的编号
I1.sort()  # 对列表进行排序

# Concatenate of representation vectors generated by five drug feature networks
E = np.zeros((841, 320), float)
for i in I1:
    #E[int(i) - 1][0:128] = model.vectors[str(i)]
    #E[int(i) - 1][0:128] = model_d[int(i) - 1]
    #E[int(i) - 1][0:128] = model.vectors[str(i)]
    E[int(i) - 1][0:64] = model.vectors[str(i)]
    E[int(i) - 1][64:128] = model_s[int(i) - 1]
    E[int(i) - 1][128:192] = model_t[int(i) - 1]
    E[int(i) - 1][192:256] = model_e[int(i) - 1]
    E[int(i) - 1][256:320] = model_p[int(i) - 1]
    #E[int(i)-1][512:640]=model_d[int(i)-1]
    #E[int(i) - 1][512:640] = model.vectors[str(i)]

# concate
#X_train1 = []
#X_train2 = []
#Y_train = []
d1=[]
d2=[]

file_namePN = 'ourdata_dd.csv'
# 创建一个空的数组，用于存储CSV文件中的数据
trainPosition = []
# 使用CSV模块打开CSV文件并读取数据
with open(file_namePN, mode='r', newline='') as file:
    reader = csv.reader(file)  # 将reader变量更正为trainPosition
    # 遍历CSV文件的每一行，并将每一行的数据作为列表添加到数组中
    for row in reader:
        trainPosition.append(row)
trainPosition = [[int(item[0]), int(item[1])] for item in trainPosition]

for i in range(0, len(trainPosition)):
    d1.append(E[trainPosition[i][0]])  # 使用[trainPosition[i][0]]作为索引
    d2.append(E[trainPosition[i][1]])  # 使用[trainPosition[i][1]]作为索引


d1 = np.array(d1)  # 将其转换成数组,边的其中一个节点的128维特征，数组（312150，640）
d2 = np.array(d2)  # 将其转换成数组，边的另一个节点的128维特征，数组（312150，640）
combined_dd = np.hstack((d1, d2))
df = pd.DataFrame(combined_dd)
df.to_csv('16w+combined_Feature_dd1_64_5_2.csv', index=False)#82620*1280
print("文件已成功保存。")


'''
# Test set
X_test1 = []
X_test2 = []
Y_test = []
# 指定要读取的CSV文件的文件名
file_nameN = 'GenerateNegativeSample.csv'
# 创建一个空的数组，用于存储CSV文件中的数据
testPosition = []
# 使用CSV模块打开CSV文件并读取数据
with open(file_nameN, mode='r', newline='') as file:
    reader = csv.reader(file)
    # 遍历CSV文件的每一行，并将每一行的数据作为列表添加到数组中
    for row in reader:
        testPosition.append(row)
testPosition = [[int(item[0]), int(item[1])] for item in testPosition]

for i in range(0, len(testPosition)):
    X_test1.append(E[testPosition[i][0]])  # 使用[trainPosition[i][0]]作为索引
    X_test2.append(E[testPosition[i][1]])  # 使用[trainPosition[i][1]]作为索引
X_test1 = np.array(X_test1)
X_test2 = np.array(X_test2)
combined_X_test = np.hstack((X_test1, X_test2))
df = pd.DataFrame(combined_X_test)
df.to_csv('combined640_Feature_Negative.csv', index=False)#82620*1280
print("CSV 文件 combined640_Feature_Negative已成功保存。")
'''
