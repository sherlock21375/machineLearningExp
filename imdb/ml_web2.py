# -*- coding: utf-8 -*-
# @Time    : 2020/1/26 16:15
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : ml_web2.py
# @Software: PyCharm
import re  # 正则表达式
from bs4 import BeautifulSoup  # html标签处理
from sklearn.externals import joblib
import flask
from flask import Flask, request, url_for, Response
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

# test = pd.read_csv('./testData.tsv', header=0, delimiter="\t", quoting=3)
# test = test[0:10]
# test = ['i think this movie is very good,wonderful','i think this movie is very bad,bitch','i like this movie,actors are very hardworking']
# test_data = []
# for i in range(0, len(test)):
#     test_data.append(" ".join(review_to_wordlist(test[i])))
# print(test_data)
# count_vect = joblib.load('count_vect')
# X_all = count_vect.transform((d for d in test_data))
# # 加载模型文件，生成模型对象
# new_model = joblib.load("model.pkl")
# print(new_model.predict(X_all))


app = Flask(__name__)

# 加载模型
model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def index():
    with app.test_request_context():
        # 生成每个函数监听的url以及该url的参数
        result = {"predict_iris": {"url": url_for("predict_iris"),
                                   "params": ["sepal_length", "sepal_width", "petal_length", "petal_width"]}}

        result_body = flask.json.dumps(result)

        return Response(result_body, mimetype="application/json")


@app.route("/ml/predict_iris", methods=["GET"])
def predict_iris():
    request_args = request.args

    # 如果没有传入参数，返回提示信息
    if not request_args:
        result = {
            "message": "请输入参数：sepal_length, sepal_width, petal_length, petal_width"
        }
        result_body = flask.json.dumps(result, ensure_ascii=False)
        return Response(result_body, mimetype="application/json")

    # 获取请求参数
    data = request_args.get("data", "i think this movie is very good,wonderful")
    test = ['i think this movie is very good,wonderful', 'i think this movie is very bad,bitch',
            'i like this movie,actors are very hardworking']
    test_data = []
    test_data.append(data)
    # for i in range(0, len(test)):
    #     test_data.append(" ".join(review_to_wordlist(test[i])))

    # 构建特征矩阵
    count_vect = joblib.load('count_vect')
    X_all = count_vect.transform((d for d in test_data))

    # 生成预测结果
    predict_result = model.predict(X_all)
    print("predict_result: {0}".format(predict_result))
    predict_result = predict_result[0]
    print(predict_result)

    # 构造返回数据
    result = {
        "features": {
            "data": data,
        },
        "result": int(predict_result)
    }

    result_body = flask.json.dumps(result, ensure_ascii=False)
    return Response(result_body, mimetype="application/json")


if __name__ == "__main__":
    app.run(port=8000)
