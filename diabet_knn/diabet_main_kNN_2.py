from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold,cross_val_score
data = pd.read_csv("./data/diabetes.csv")

# 分离特征与目标
X = data.iloc[:, 0:8]
Y = data.iloc[:, 8]
# print('shape of X {};shape of Y {}'.format(X.shape, Y.shape))

# 划分训练集与测试集
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

# 模型比较，普通的k均值算法、带权重的k均值算法、指定半径的k均值算法
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

# 构造三个模型
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=9)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=9, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(n_neighbors=9, radius=500.0)))

results = []
for name,model in models:
    # 10折
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model,X,Y,cv=kfold)
    results.append((name,cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {:.2f}'.format(results[i][0],results[i][1].mean()))

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
train_score=knn.score(X_train,y_train)
test_score = knn.score(X_test,y_test)
print('train score: {:.2f}; test score: {:.2f}'.format(train_score,test_score))

# ShuffleSplit对数据集打乱再分配
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y,ylim=None, cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# plot_learning_curve(knn, 'Learning Curve for KNN Diabetes', X, Y,ylim=(0.0, 1.01), cv=cv)
# plt.show()



from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=2)
X_new = selector.fit_transform(X, Y)
# print(X_new[0:5])
plt.figure(figsize=(8, 5), dpi=100)
plt.ylabel('BMI')
plt.xlabel('Glucose')
# 画出Y==0的阴性样本，用圆圈表示
plt.scatter(X_new[Y == 0][:, 0], X_new[Y == 0][:, 1], c='r', marker='o', s=10)
# 画出Y==1的阳性样本，用三角形表示
plt.scatter(X_new[Y == 1][:, 0], X_new[Y == 1][:, 1], c='g', marker='^', s=10)
plt.show()