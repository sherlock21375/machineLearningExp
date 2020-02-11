from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv("./data/diabetes.csv")

#------------1--------------------
# print(diabetes.columns)  # check the column names
# print(diabetes.head())     # default n = 5，show five rows
# # the results 0 means not having diabetes,1 means having diabetes
# # get the cows and cols of diabetes
# print("dimension of diabetes data: {}".format(diabetes.shape))
# # group the "Outcome" data to get the number of each group
# print(diabetes.groupby('Outcome').size())

#-------------2---------------------
# print(diabetes.info())   #get the diatebes' info 可不加print
# print(diabetes["Outcome"].describe())


#-------------3-----------------
# sns.countplot(diabetes['Outcome'], label="Count")
# # plt.savefig("糖尿病数据处理图片/0_1_graph")
# plt.show()

#-------------4------------------
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.loc[:, diabetes.columns != 'Outcome'],
    diabetes['Outcome'], stratify=diabetes['Outcome'],
    random_state=66)


#-------------5-------------------------
from sklearn.neighbors import KNeighborsClassifier

# training_accuracy = []
# test_accuracy = []
# best_prediction=[-1,-1]
# # try n_neighbors from 1 to 10
# neighbors_settings = range(1, 11)
# for n_neighbors in neighbors_settings:
#     knn = KNeighborsClassifier(n_neighbors=n_neighbors)  # build the models
#     knn.fit(x_train, y_train)  # use x_train as train data and y_train as target value
#     training_accuracy.append(knn.score(x_train, y_train))  # record training set accuracy
#     if training_accuracy > best_prediction[n_neighbors]:
#       best_prediction=[n_neighbors,training_accuracy]
#     test_accuracy.append(knn.score(x_test, y_test))  # record test set accuracy
#     if test_accuracy > best_prediction[n_neighbors]:
#       best_prediction=[n_neighbors,test_accuracy]
# '''
# The relationship between the training set and the test set on the model prediction
# accuracy (Y-axis) and the number of nearest neighbors (X-axis) is demonstrated
# '''
# plt.figure()
# plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
# plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("n_neighbors")
# plt.legend()
# # plt.savefig("糖尿病数据处理图片/knn_compare_model")
#print(best_prediction)
# plt.show() #显示不同k值的精确度

# select n_neighbors = 9
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)
print("Accuracy of K-NN classifier on training set: {:.2f}".format(knn.score(x_train, y_train)))
print("Accuracy of K-NN classifier on test set: {:.2f}".format(knn.score(x_test, y_test)))

