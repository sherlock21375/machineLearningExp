# -*- coding: utf-8 -*-
# @Time    : 2020/1/27 9:30
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : main.py
# @Software: PyCharm
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import jieba as jb
import re

pd.set_option('expand_frame_repr', False) #数据超过总宽度后，是否折叠显示

df = pd.read_csv('./data/shopping.csv')
df = df[['cat', 'review']]
print("数据总量: %d ." % len(df))
print("----------样例数据----------")
print(df.sample(10))
# print(df.describe())
print("----------显示空值个数----------")
print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
df[df.isnull().values==True]#只显示含有缺失值的列
df = df[pd.notnull(df['review'])]#删除含有缺失值的列
print("----------显示删除空值位后附近的数值----------")
print(df.iloc[36420:36430])
print("----------显示空值个数----------")
print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
d = {'cat':df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
df_cat = pd.DataFrame(data=d).reset_index(drop=True)
print(df_cat)

# df_cat.plot(x='cat', y='count', kind='bar', legend=False,  figsize=(8, 5))
# plt.rcParams['font.sans-serif'] = ['SimHei'] #显示中文
# plt.title("类目数量分布")
# plt.ylabel('数量', fontsize=18)
# plt.xlabel('类目', fontsize=18)
# # plt.show()

df['cat_id'] = df['cat'].factorize()[0]
cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)
cat_to_id = dict(cat_id_df.values)
id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)
print("----------显示增加cat_id后情形----------")
print(cat_id_df.sample(10))

# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

# 加载停用词
stopwords = stopwordslist("./data/chineseStopWords.txt")
#删除除字母,数字，汉字以外的所有符号
df['clean_review'] = df['review'].apply(remove_punctuation)
print("----------显示删除标点符号等后情形----------")
print(df.sample(10))
#分词，并过滤停用词
#在clean_review的基础上进行分词,把每个评论内容分成由空格隔开的一个一个单独的词语。
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))
print("----------显示分词后情形----------")
print(df.head())

from collections import Counter
from wordcloud import WordCloud


def generate_wordcloud(tup):
    wordcloud = WordCloud(background_color='white',
                          font_path='simhei.ttf',
                          max_words=50, max_font_size=40,
                          random_state=42
                          ).generate(str(tup))
    return wordcloud


cat_desc = dict()
for cat in cat_id_df.cat.values:
    text = df.loc[df['cat'] == cat, 'cut_review']
    text = (' '.join(map(str, text))).split(' ')
    cat_desc[cat] = text

# fig, axes = plt.subplots(5, 2, figsize=(30, 38))
# k = 0
# for i in range(5):
#     for j in range(2):
#         cat = id_to_cat[k]
#         most100 = Counter(cat_desc[cat]).most_common(100)
#         ax = axes[i, j]
#         ax.imshow(generate_wordcloud(most100), interpolation="bilinear")
#         ax.axis('off')
#         ax.set_title("{} Top 100".format(cat), fontsize=10)
#         k += 1
# # plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
features = tfidf.fit_transform(df.cut_review)
labels = df.cat_id
print(features.shape)
print('------------显示向量化后结果-----------------')
print(features)

from sklearn.feature_selection import chi2
import numpy as np

N = 2
for cat, cat_id in sorted(cat_to_id.items()):
    features_chi2 = chi2(features, labels == cat_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(cat))
    print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['cut_review'], df['cat_id'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

def myPredict(sec):
    format_sec=" ".join([w for w in list(jb.cut(remove_punctuation(sec))) if w not in stopwords])
    pred_cat_id=clf.predict(count_vect.transform([format_sec]))
    print(id_to_cat[pred_cat_id[0]])

myPredict("性能还是不错的")



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
# 箱体图上可以看出随机森林分类器的准确率是最低的
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns
#
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()

print(cv_df.groupby('model_name').accuracy.mean())

# 针对平均准确率最高的LinearSVC模型，我们将查看混淆矩阵，并显示预测标签和实际标签之间的差异
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 训练模型
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.33, stratify=labels,
                                                                                 random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 生成混淆矩阵
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=cat_id_df.cat.values, yticklabels=cat_id_df.cat.values)
plt.ylabel('实际结果', fontsize=18)
plt.xlabel('预测结果', fontsize=18)
plt.show()

from sklearn.metrics import classification_report

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=cat_id_df['cat'].values))