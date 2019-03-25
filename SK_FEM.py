# coding=gbk
'''
Created on 2017��5��7��

@author: yuziqi

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# ����Ԥ��FEM����
input_data_path = "./data/input_data.csv"

df_read = pd.read_csv(input_data_path, header=None, names=['number', 'BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3'])

# �������ݹ�ģ
# print(boston.data.shape)
print('df_read:', df_read.shape)


# ���Ū�����������ĺ���Ҳ��һ����ϰ��
# print(boston.feature_names)
# print(boston.DESCR)

# �����һ�����裬���������Ƿ����滯��һ�㶼��û�е�
import numpy as np

# print(np.max(boston.target))
# print(np.min(boston.target))
# print(np.mean(boston.target))

from sklearn.cross_validation import train_test_split

# 1.�����ݽ��зָ�
data = df_read[['BB0', 'BB1', 'F0', 'F1', 'F2', 'F3', 'HH0', 'HH1', 'P0', 'P1', 'P2', 'P3']]
target = df_read['number']
# X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler

# 2.�ֱ��X��y���滯�����滯��Ŀ�����ڱ���ԭʼ����ֵ������󣬵���ѵ���õ��Ĳ���Ȩ�ز�һ
scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

# scalery = StandardScaler().fit(y_train)
scalery = StandardScaler().fit(y_train.values.reshape(-1, 1)) # ���°汾�����ж�����������һ��2D���󣬼�ʹ��һ���򵥵�column��row��
# y_train = scalery.transform(y_train)
y_train = scalery.transform(y_train.values.reshape(-1, 1))
y_test = scalery.transform(y_test.values.reshape(-1, 1))

# 3.�Ȱ�����ģ��д�ã���Ȼ��Ĭ��5�۽�����֤��ֻ�����������ָ�겻���Ǿ��ȣ�������һ������R2�������ϣ�����÷ֶ��ٴ����ж��ٷֱȵĻع������Ա�ѵ�������Ǻͽ���
from sklearn.cross_validation import *


def train_and_evaluate(clf, X_train, y_train):
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))


# 4.�ع�ģ�ͣ��Ƚ��д����Ե���3��

# 4.1 ��������ģ�ͳ��ԣ� SGD_Regressor
from sklearn import linear_model

# ������һ�����򻯵�ѡ��penalty��Ŀǰ14ά����Ҳ������̫��Ӱ��
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty=None, random_state=42)
train_and_evaluate(clf_sgd, X_train, y_train)
# print(clf_sgd.score(X_test, y_test))
clf_sgd.fit(X_train, y_train)
print('clf_sgd', clf_sgd.score(X_test, y_test))

# 4.2 �ٻ�һ��SGD_Regressor��penalty����Ϊl2,���ò��Ӱ�첻����Ϊ����̫�٣��������岻��
clf_sgd_l2 = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', random_state=42)
train_and_evaluate(clf_sgd_l2, X_train, y_train)
# print(clf_sgd_l2.score(X_test, y_test))
clf_sgd_l2.fit(X_train, y_train)
print('clf_sgd_l2', clf_sgd_l2.score(X_test, y_test))

# 4.3 �ٿ���SVM��regressor��ô��������Ĭ�ϲ�����,
from sklearn.svm import SVR

# ʹ�����Ժ�û��ɶ��������������Ϊ�����٣����Կ��Կ�������ά��
clf_svr = SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)
# print(clf_svr.score(X_test, y_test))
clf_svr.fit(X_train, y_train)
print('clf_svr', clf_svr.score(X_test, y_test))

# 4.4 ����ά�ȣ�Ч�����ԣ����Ǵ�������@@�������ߵĻ�, CPU�����ܲ��ˣ��ڴ浹��С�¡���ʵ�������ڣ��������Լ���û�취ֱ�ӽ�����Щ�����ľ��庬���ˡ�
clf_svr_poly = SVR(kernel='poly')
train_and_evaluate(clf_svr_poly, X_train, y_train)
# print(clf_svr_poly.score(X_test, y_test))
clf_svr_poly.fit(X_train, y_train)
print('clf_svr_poly', clf_svr_poly.score(X_test, y_test))

# 4.5 RBF (������˸���ţ�ƣ�)
clf_svr_rbf = SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train)
# print(clf_svr_rbf.score(X_test, y_test))
clf_svr_rbf.fit(X_train, y_train)
print('clf_svr_rbf', clf_svr_rbf.score(X_test, y_test))

# 4.6 ���������͵�! ���޻ع�ɭ�֣��Ŵ����ˣ�����
from sklearn import ensemble

clf_et = ensemble.ExtraTreesRegressor()
train_and_evaluate(clf_et, X_train, y_train)

# ��󿴿��ڲ��Լ��ϵı���
clf_et.fit(X_train, y_train)
# print(clf_et.predict(X_test))
# print('y_test:', y_test)
print('clf_et', clf_et.score(X_test, y_test))
