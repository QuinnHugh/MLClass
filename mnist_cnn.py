"""CNN 网络对MNIST分类"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

#设置cuda device
if torch.cuda.is_available():
    device = torch.device('cuda', 0)
else:
    device = torch.device('cpu')

#设置训练参数
EPOCHS = 200
BATCH_SIZE = 100
LR = 0.0001
DOWNLOAD_MNIST = True

#获取mnist数据集
train_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

#数据集显示
print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

#dataloader载入训练数据集以及测试数据集
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=5
)

test_data = torchvision.datasets.MNIST(
    root="./mnist",
    train=False
)

test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatitle=True).type(torch.FloatTensor)/255.
test_y = test_data.test_labels

##搭建cnn网络
class CNN(nn.Module):
    """CNN网络"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )#第一层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )#第二层卷积
        self.fc = nn.Linear(64*7*7, 128)
        self.out = nn.Linear(128,10)

    def forward(self, image):
        """前向传播"""
        image = self.conv1(image)
        image = self.conv2(image)
        fc = image.view(image.size(0), -1)
        fc = self.fc(fc)
        result = self.out(fc)
        return result

cnn = CNN().to(device)
print(cnn)

optimizer = torch.optim.adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x).to(device)
        b_y = Variable(y).to(device)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            test_output = cnn(test_x)
            predict = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = 1 - float(torch.nonzero(predict - test_y).shape[0]) / float(test_y.size(0))
            print("Epoch:{}  train loss {}, accuracy {}".format(epoch, loss, accuracy))

test_output = cnn(test_x[:50])
predict = torch.max(test_output, 1)[1].data.squeeze()
print("predict ", test_output)
print("real ", test_y[:50])



