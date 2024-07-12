# 在这个代码中，我们生成了一个包含10个特征的分类数据集。然后，我们定义了两个分类器（逻辑回归和决策树），并将它们存储在字典中以便我们可以遍历其中的所有分类器。同时，我们还使用StratifiedKFold创建了一个5折交叉验证。
#
# 在计算每个分类器的ROC曲线和平均AUC值时，我们遍历了交叉验证的每一折，并计算每次交叉验证中的ROC曲线和AUC值。我们使用插值的方法将每个模型的ROC曲线归一化到一致的FPR轴上，并计算平均ROC曲线和平均AUC值。
#
# 最后，我们将所有ROC曲线绘制在同一张图中，并添加相应的标签和标题，以便我们可以比较不同分类器的性能。

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from RotationForest import *


# 生成分类数据，并将其分为训练集和测试集
# X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
train_features = 'C://Users//wsco45//Desktop//test_code_CDA//1_circRNA-disease//0_testHGLCDA//HGLCDA//five_fold_train_features.mat'
train_labels = 'C://Users//wsco45//Desktop//test_code_CDA//1_circRNA-disease//0_testHGLCDA//HGLCDA//five_fold_train_labels.mat'
test_features = 'C://Users//wsco45//Desktop//test_code_CDA//1_circRNA-disease//0_testHGLCDA//HGLCDA//five_fold_test_features.mat'
test_labels = 'C://Users//wsco45//Desktop//test_code_CDA//1_circRNA-disease//0_testHGLCDA//HGLCDA//five_fold_test_labels.mat'
Train_data = scio.loadmat(train_features)
Train_labels = scio.loadmat(train_labels)
Test_data = scio.loadmat(test_features)
Test_labels = scio.loadmat(test_labels)

five_fold_train_features = Train_data['five_fold_train_features']
five_fold_train_labels = Train_labels['five_fold_train_labels']
five_fold_test_features = Test_data['five_fold_test_features']
five_fold_test_labels = Test_labels['five_fold_test_labels']

train_data = []
train_labels = []
test_data = []
test_labels = []
for i in five_fold_train_features:
    train_data.append(i)
for i in five_fold_train_labels:
    train_labels.append(i)
for i in five_fold_test_features:
    test_data.append(i)
for i in five_fold_test_labels:
    test_labels.append(i)
print(train_data,train_labels,test_data,test_labels)
print("### Different Classifiers Compare of 5Fold cross-validation:")

# 定义不同分类器和5折交叉验证

lr_clf = LogisticRegression(penalty="l2",solver="liblinear",C=0.1,max_iter=1000)
knn_clf = KNeighborsClassifier(n_neighbors=5)
nb_clf = GaussianNB()
dt_clf = DecisionTreeClassifier(random_state=42)
svm_clf = SVC(kernel='poly', degree=10, gamma=1, coef0=1, probability=True) #kernel='linear'; kernel='poly', degree=3, gamma=0.1, coef0=0; kernel='rbf', gamma=0.1
sgd_clf = SGDClassifier(loss='log', max_iter=1000, alpha=0.01,shuffle=True,random_state=42)
rf_clf = RandomForestClassifier(random_state=42,n_estimators=100,min_samples_split=2,min_samples_leaf=21)
rof_clf = RotationForest(n_classifiers=16,K = 41)

# cv = StratifiedKFold(n_splits=5)
# 使用交叉验证计算模型的ROC曲线
# 定义字典存储分类器对象和对应的标签
classifiers = {'RoF': rof_clf, 'RF': rf_clf, 'LR': lr_clf, 'KNN': knn_clf, 'NB': nb_clf, 'DT': dt_clf, 'SVM': svm_clf, 'SGD': sgd_clf }

# 遍历所有分类器，计算ROC曲线和AUC值
for key, classifier in classifiers.items():
    tprs, aucs, probas = [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    # 5折交叉验证
    # for i, (train, test) in enumerate(cv.split(X, y)):
    for i in range(5):
        classifier.fit(train_data[i], train_labels[i])
        probas_ = classifier.predict_proba(test_data[i])
        probas.append(probas_)
        fpr, tpr, thresholds = roc_curve(test_labels[i], probas_[:,1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr) #计算AUC
        aucs.append(roc_auc)

    # 绘制ROC曲线，计算平均AUC值
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, label='{} (AUC = {:0.4f} $\pm$ {:0.4f})'.format(key, mean_auc, std_auc), lw=1.5, alpha=1)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
          alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='lower right')
plt.savefig('./Figures/'+ 'DifferentClassifiersCompare.png')
plt.show()
