"""
线性分类器interface
10分类 采用OvR的方法来训练
"""
import torch as tc

import loadmnist as lm
import linage_classify

#载入MNIST数据集
img, label = lm.load_mnist("./data", "train", 'tc')
test, tl = lm.load_mnist("./data", "t10k", 'tc')

#把MNIST数据集的标签分类10个二分类器,0为正类,1为负类
filter_vec = tc.tensor(range(10))
filter_mat = filter_vec.repeat(label.size()[0], 1).byte()
label_mat = tc.clamp(label.repeat(10, 1) - filter_mat.t(), 0, 1)

#初始化线性分类器,使用最小二乘法类
lcm = linage_classify.LeastSquares()
lcm.cuda()##使用cuda

#训练所有的10个分类器并将结果存入y_hat中
y_hat = tc.Tensor(10, tl.size()[0])
for i in range(10):
    lcm.train(img, label_mat[i])
    y_hat[i] = lcm.predict(test).abs()

#取y_hat中的最接近正类(0)的结果为分类结果
_, y = tc.min(y_hat.t(), 1)

#错误统计
err = tc.sum(tc.clamp((y.cpu()-tl.long()).abs(), 0, 1))
rate = err.item()/tl.size()[0]

print(err.item(), 1-rate)##输出错误个数和正确率
