# coding=gbk
'''
Created on 2017年5月7日

@author: yuziqi

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# 首先预读FEM数据
input_data_path = "./data/input_data.csv"

df_read = pd.read_csv(input_data_path, header=None, names=['number', 'BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3'])

# 查验数据规模
# print(boston.data.shape)
print('df_read:', df_read.shape)


# 多多弄懂数据特征的含义也是一个好习惯
# print(boston.feature_names)
# print(boston.DESCR)

# 这里多一个步骤，查验数据是否正规化，一般都是没有的
import numpy as np

# print(np.max(boston.target))
# print(np.min(boston.target))
# print(np.mean(boston.target))

from sklearn.cross_validation import train_test_split

# 1.对数据进行分割
data = df_read[['BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3']]
target = df_read['number']
# X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

# 2.分别对X，y正规化：正规化的目的在于避免原始特征值差异过大，导致训练得到的参数权重不一
scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

# scalery = StandardScaler().fit(y_train)
scalery = StandardScaler().fit(y_train.values.reshape(-1, 1)) # （新版本中所有东西都必须是一个2D矩阵，即使是一个简单的column或row）
# y_train = scalery.transform(y_train)
y_train = scalery.transform(y_train.values.reshape(-1, 1))
y_test = scalery.transform(y_test.values.reshape(-1, 1))

# 3.先把评价模块写好，依然是默认5折交叉验证，只是这里的评价指标不再是精度，而是另一个函数R2，大体上，这个得分多少代表有多大百分比的回归结果可以被训练器覆盖和解释
from sklearn.cross_validation import *


def train_and_evaluate(clf, X_train, y_train):
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))


# 4.回归模型，比较有代表性的有3种

# 4.1 先用线性模型尝试， SGD_Regressor
from sklearn import linear_model

# 这里有一个正则化的选项penalty，目前14维特征也许不会有太大影响
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train)
# print(clf_sgd.score(X_test, y_test))
clf_sgd.fit(X_train, y_train)
print('clf_sgd', clf_sgd.score(X_test, y_test))

# 4.2 再换一个SGD_Regressor的penalty参数为l2,结果貌似影响不大，因为特征太少，正则化意义不大
clf_sgd_l2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd_l2, X_train, y_train)
# print(clf_sgd_l2.score(X_test, y_test))
clf_sgd_l2.fit(X_train, y_train)
print('clf_sgd_l2', clf_sgd_l2.score(X_test, y_test))

# 4.3 再看看SVM的regressor怎么样（都是默认参数）,
from sklearn.svm import SVR

# 使用线性核没有啥子提升，但是因为特征少，所以可以考虑升高维度
clf_svr = SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)
# print(clf_svr.score(X_test, y_test))
clf_svr.fit(X_train, y_train)
print('clf_svr', clf_svr.score(X_test, y_test))

# 4.4 升高维度，效果明显，但是此招慎用@@，特征高的话, CPU还是受不了，内存倒是小事。其实到了现在，连我们自己都没办法直接解释这些特征的具体含义了。
clf_svr_poly = SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train)
# print(clf_svr_poly.score(X_test, y_test))
clf_svr_poly.fit(X_train, y_train)
print('clf_svr_poly', clf_svr_poly.score(X_test, y_test))

# 4.5 RBF (径向基核更是牛逼！)
clf_svr_rbf = SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train)
# print(clf_svr_rbf.score(X_test, y_test))
clf_svr_rbf.fit(X_train, y_train)
print('clf_svr_rbf', clf_svr_rbf.score(X_test, y_test))

# 4.6 再来个更猛的! 极限回归森林，放大招了！！！
from sklearn import ensemble

clf_et = ensemble.ExtraTreesRegressor()
train_and_evaluate(clf_et, X_train, y_train)

# 最后看看在测试集上的表现
clf_et.fit(X_train, y_train)
# print(clf_et.predict(X_test))
# print('y_test:', y_test)
print('clf_et', clf_et.score(X_test, y_test))
