# -*- coding: utf-8 -*-
# @Time    : 2020/1/26 16:00
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : 2_test.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 加载鸢尾花数据
iris = load_iris()
# 创建包含特征名称的 DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 生成标记，切分训练集、测试集
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
train, test = df[df['is_train']==True], df[df['is_train']==False]

# 生成 X 和 y
features = df.columns[:4]
y = pd.factorize(train['species'])[0]

model = RandomForestClassifier(n_jobs=2)

# 训练模型
model.fit(train[features], y)
# 预测数据
model.predict(test[features])

from sklearn.externals import joblib

# 保存模型到 model.joblib 文件
joblib.dump(model, "model_iris.joblib" ,compress=1)

# 加载模型文件，生成模型对象
new_model = joblib.load("model_iris.joblib")

new_pred_data = [[0.5, 0.4, 0.7, 0.1]]
# 使用加载生成的模型预测新样本
new_model.predict(new_pred_data)