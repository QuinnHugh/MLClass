"""
分类器汇总
"""
import torch as tc

class CudaSupport(object):
    """cuda支持"""
    device = tc.device('cpu')##用于设置计算设备,默认cpu
    def __init__(self):
        ##默认初始化cuda
        self.cuda()

    def cuda(self, device_index=0):
        """
        设置类的cuda支持,更新类内的device的值
        参数列表:
        device_index 在有多张显卡的时候设置cuda设备,默认为第0个cuda设备
        """
        if tc.cuda.is_available():
            self.device = tc.device('cuda', device_index)
        else:
            self.device = tc.device('cpu')
            print("\n Warning:设备不支持cuda\n")



class LeastSquares(CudaSupport):
    """
    带Redige正则项的最小二乘法
    注:alpha不要设置为0
    可以使用cuda
    使用train方法来更新w,
    predict方法给出预测值
    """
    w_hat = None
    xs = None
    ys = None
    def __init__(self, alpha=0.2):
        CudaSupport.__init__(self)
        #设置正则项的权值
        self.alpha = alpha

    def train(self, train, labels):
        """
        写入训练样本并且更新w_hat
        参数列表:
        train 类型tensor为训练数据
        labels 类型tensor为训练标签
        """
        self.xs = train.to(self.device)
        self.ys = (labels.float()).to(self.device)
        self.update_w(self.alpha)


    def _w(self, alpha):
        """
        通过最小二乘法计算w的值,并更新类内的w_hat
        参数列表:
        alpha 为Ridge回归的参数
        """
        vector_1 = tc.ones(self.ys.size()[0], 1).to(self.device)
        x_mat = tc.cat((self.xs.float(), vector_1), 1)##将输入的x变为增广矩阵
        #书上公式\hat{w} = (X^TX)^{-1}X^T\vec{y}
        xtx = tc.mm(x_mat.t(), x_mat)
        regression_mat = (tc.eye(xtx.size()[0]) * alpha).to(self.device)
        temp = tc.mm((xtx - regression_mat).inverse(), x_mat.t())
        #结果转为行向量写入w_hat
        self.w_hat = temp.mm(self.ys.view(self.ys.size()[0], 1))

    def update_w(self, alpha):
        """
        更新w值,用于被显示调用
        参数列表:
        alpha 类似于_w()
        """
        self._w(alpha)

    def predict(self, test_x):
        """
        输入测试数据根据训练得到的w_hat给出预测值
        参数列表:
        test_x 类型为tensor,即测试数据
        返回值:
        为预测值tensor
        """
        #输入变成增广矩阵
        vector_1 = tc.ones(test_x.size()[0], 1).to(self.device)
        x_mat = tc.cat((test_x.to(self.device).float(), vector_1), 1)
        if self.w_hat is None:
            self.update_w(self.alpha)##若w为空则更新w
        return tc.mm(x_mat, self.w_hat).view(test_x.size()[0])


class FisherClassify(CudaSupport):
    """费舍尔分类器LDA"""
    w_hat = None
    thresh = None
    x_1 = None
    x_2 = None

    u_1 = None
    u_2 = None
    Sw_1 = None
    Sw_2 = None

    def __init__(self):
        """初始化分类器参数"""
        CudaSupport.__init__(self)

    def train(self, x_1, x_2):
        """
        用于接收训练数据
        参数x_1: tensor
        参数x_2: tensor
        """
        self.x_1 = x_1.to(self.device)
        self.x_2 = x_2.to(self.device)
        self.update_w(self.x_1.float(), self.x_2.float())

    def _calculate(self, t):
        """
        输入一个tensor的类数据计算类的均值向量以及类内离散矩阵
        参数tensor: 类数据tensor
        返回u:类均值向量
        返回S_w:类内离散矩阵
        """
        u = t.mean(0)
        u_mat = u.repeat(t.size()[0], 1)
        sub_mat = t - u_mat
        S_w = tc.mm(sub_mat.t(), sub_mat)
        return u, S_w

    def _w(self):
        S_w = self.Sw_1 + self.Sw_1
        self.w_hat = tc.mv(S_w.inverse(), self.u_2 - self.u_1)

    def _cal_thresh(self):
        """计算用于判断分类的界限值"""
        center_1 = tc.dot(self.w_hat, self.u_1)
        center_2 = tc.dot(self.w_hat, self.u_2)
        self.thresh = (center_1 + center_2) / 2

    def update_w(self, x_1, x_2):
        """
        更新w值,用于被显示调用
        参数x_1: tensor
        参数x_2: tensor
        """
        self.Sw_1, self.u_1 = self._calculate(x_1)
        self.Sw_2, self.u_2 = self._calculate(x_2)
        self._w()

    def predict(self, test):
        """
        用于利用测试数据预测结果
        """
        #将测试结果分为正和负两类(离边界越远越可信度强)
        cuda_test = test.to(self.device).float()
        y_hat = tc.mm(cuda_test, self.w_hat) - self.thresh.repeat(test.shape[0], 1)
        return y_hat


class PerceptronClassify(CudaSupport):
    """
    基于感知机的分类器
    """
    def __init__(self):
        CudaSupport.__init__(self)
