# -*- coding: utf-8 -*-
# @Time    : 2020/1/26 14:06
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : 1.py
# @Software: PyCharm
import re  # 正则表达式
from bs4 import BeautifulSoup  # html标签处理
import pandas as pd


def review_to_wordlist(review):
    '''
    把IMDB的评论转成词序列
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review,features="html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words


# 使用pandas读入训练和测试csv文件
train = pd.read_csv('./labeledTrainData.tsv', header=0, delimiter="\t",quoting=3)
print(train.head())
test = pd.read_csv('./testData.tsv', header=0, delimiter="\t", quoting=3)
print(test.head())
# 取出情感标签，positive/褒 或者 negative/贬
y_train = train['sentiment']
# 将训练和测试数据都转成词list
train_data = []
for i in range(0, len(train['review'])):
    train_data.append(" ".join(review_to_wordlist(train['review'][i])))
# print(train_data)
test_data = []
for i in range(0, len(test['review'])):
    test_data.append(" ".join(review_to_wordlist(test['review'][i])))
# print(test_data[0])

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
# 初始化TFIV对象，去停用词，加2元语言模型
tfv = TFIV(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
           ngram_range=(1, 2), use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words='english')
# 合并训练和测试集以便进行TFIDF向量化操作
X_all = train_data + test_data
len_train = len(train_data)

# 这一步有点慢，去喝杯茶刷会儿微博知乎歇会儿...
# tfv.fit(X_all)
X_all = tfv.fit_transform(X_all)
# 恢复成训练集和测试集部分
X = X_all[:len_train]
X_test = X_all[len_train:]
print("--------X_test--------")
print(X_test[0])


# 多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB as MNB
import numpy as np
model_NB = MNB()
model_NB.fit(X, y_train)  # 特征数据直接灌进来
MNB(alpha=1.0, class_prior=None, fit_prior=True)

from sklearn.model_selection import KFold,cross_val_score

kfold = KFold(n_splits=20)
print("多项式贝叶斯分类器20折交叉验证得分: ",np.mean(cross_val_score(model_NB,X,y_train,cv=kfold,scoring='roc_auc')))
# 多项式贝叶斯分类器20折交叉验证得分:  0.9497168922054018
print(model_NB.predict(X_test))
from sklearn.externals import joblib
# 保存模型到 model.joblib 文件
joblib.dump(model_NB, "model.pkl" ,compress=1)
joblib.dump(tfv, 'count_vect')
# # 加载模型文件，生成模型对象
# new_model = joblib.load("model.joblib")
# new_pred_data = [[0.5, 0.4, 0.7, 0.1]]
# # 使用加载生成的模型预测新样本
# new_model.predict(new_pred_data)


# test = test[0:10]
# test_data = []
# for i in range(0, len(test['review'])):
#     test_data.append(" ".join(review_to_wordlist(test['review'][i])))
# X_all = tfv.transform((d for d in test_data))
# print(model_NB.predict(X_all))
