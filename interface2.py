"""
线性分类器interface
10分类 采用OvR的方法来训练
fisher
"""
import torch as tc

import loadmnist as lm
import linage_classify
import tctools

#载入MNIST数据集
img, label = lm.load_mnist(".", "train", 'tc')
test, tl = lm.load_mnist(".", "t10k", 'tc')

#把MNIST数据集的标签分类10个二分类器,0为正类,1为负类
filter_vec = tc.tensor(range(10))
filter_mat = filter_vec.repeat(label.size()[0], 1).byte()
label_mat = tc.clamp(label.repeat(10, 1) - filter_mat.t(), 0, 1).cuda()

#初始化线性分类器,fisher分类器
lcm = linage_classify.FisherClassify()
lcm.cuda()##使用cuda

#训练所有的10个分类器并将结果存入y_hat中
y_hat = tc.Tensor(10, tl.size()[0]).cuda()
for i in range(10):
    c_1 = None
    c_2 = None
    for j in range(label_mat[i].size()[0]):
        if label_mat[i][j] == 0:
            c_2 = tctools.append(c_2, img[j])
        else:
            c_1 = tctools.append(c_1, img[j])
    lcm.train(c_1, c_2)
    y_hat[i] = lcm.predict(test)

#取y_hat中的最正类的结果为分类结果
_, y = tc.max(y_hat.t(), 1)

#错误统计
err = tc.sum(tc.clamp((y.cpu()-tl.long()).abs(), 0, 1))
rate = err.item()/tl.size()[0]

print(err.item(), 1-rate)##输出错误个数和正确率
