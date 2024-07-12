import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
import csv


from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, \
    accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def calculate_metrics(y_true, y_pred_prob):
    auc = roc_auc_score(y_true, y_pred_prob)
    aupr = average_precision_score(y_true, y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return auc, aupr, precision, recall, f1, accuracy


def plot_roc_and_pr(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)


    plt.figure(figsize=(12, 6))


    # ROC曲线
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')


    # PR曲线
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')


    plt.tight_layout()
    plt.show()


def main():
    # 读取特征和标签数据
    features = pd.read_csv('D:\StudyFile\Ours1\FeatureLink\combined_Feature_dd1_64_5_2.csv')
    labels = pd.read_csv('combined_Label.csv')
    results=[]
    # 使用StratifiedKFold进行五折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(features, labels):
        X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=1000,random_state=42)
        clf.fit(X_train, y_train)

        # 使用分类器进行预测
        y_pred_prob = clf.predict_proba(X_test)[:, 1]  # 获取正类别的预测概率
        # 在这里训练你的分类模型，y_pred_prob是模型的预测概率或分数

        # 计算六个指标
        results = calculate_metrics(y_test, y_pred_prob)

        # 打印或保存指标的值
        print("Results:", results)

        # 画出ROC和PR曲线
        #plot_roc_and_pr(y_test, y_pred_prob)


if __name__ == "__main__":
    main()
