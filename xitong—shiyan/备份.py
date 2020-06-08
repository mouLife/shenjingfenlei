import wx
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
# 切割图片
import cv2 as cv
from skimage import io, transform, img_as_float
import matplotlib.pyplot as plt
def ReadImageAndSplitByBian(path,bian):

    imgs = cv.imread('yuan.png',0)
    img = img_as_float(imgs)
    h,w = img.shape
    ax1 = int(h/28)-3
    ax2=int(w/28)-3
    a = [[[0] * 28 for _ in range(28)] for _ in range(ax1*ax2)]
    a_ = [[[0] * 28 for _ in range(28)] for _ in range(ax1 * ax2)]
    index=0
    label =[]
    for i in range(ax1):
        for j in range(ax2):
            b = img[i*28:(i+1)*28,j*28:(j+1)*28]
            b = np.array(b)
            b_ = imgs[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28]
            b_ = np.array(b_)
            a[index] = b
            a_[index] = b_
            label.append(index)
            index = index+1

    a = np.array(a)
    te = np.reshape(a,(ax1*ax2,28,28))

    a_ = np.array(a_)
    te_ = np.reshape(a_, (ax1 * ax2, 28, 28))
    label = np.array(label)
#    label = np.reshape(label,(ax1*ax2,5))
    te = torch.from_numpy(te)
    label = torch.from_numpy(label)
    torch_dataset = Data.TensorDataset(te, label)
    loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=torch_dataset,
        batch_size=3,
        shuffle=True,
        pin_memory=True
    )
    return loader,index,te_,imgs


def show_batch(loader):
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
class Net(nn.Module):
    def __init__(self,index):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 28 * 28 -> (28+1-5) 24 * 24
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 20 * 20
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500,index)

    def forward(self, x):
        # x: 1 * 28 * 28
        x = self.conv1(x)
        x = F.relu(x)  # 20 * 24 * 24
        x = F.max_pool2d(x, 2, 2)  # 12 * 12
        x = F.relu(self.conv2(x))  # 8 * 8
        x = F.max_pool2d(x, 2, 2)  # 4 * 4
        x = x.view(-1, 4 * 4 * 50)  # reshape (5 * 2 * 10), view(5, 20) -> (5 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return x
        return F.softmax(x, dim=1)  #


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data = Variable(torch.unsqueeze(data, dim=1).float(), requires_grad=False)
        data, target = data.to(device), target.to(device)
        target = target.long()
        pred = model(data)  # batch_size * 10
       # target = Variable(torch.unsqueeze(pred, dim=1).long(), requires_grad=False)
        loss = F.nll_loss(pred, target)
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("Train Epoch: {}, iteration: {}, Loss: {}".format(
        #         epoch, idx, loss.item()))
def Gen(image):
    img = [0 for i in range(256)]
    wide = image.shape[0]
    hight = image.shape[1]
    for i in range(wide):
        for j in range(hight):
            gray = image[i, j]
            img[gray] += 1
    y = img
    x = [i for i in range(256)]
    return x,y
def Equalize(a,image):
    b = [0 for i in range(256)]
    c = [0 for i in range(256)]
    w = image.shape[0]
    h = image.shape[1]
    mn = w * h * 1.0
    img = np.zeros([w, h], np.uint8)
    for i in range(len(a)):
        b[i] = a[i] / mn
    for i in range(len(c)):
        if i == 1:
            c[i] = b[i]
        else:
            c[i] = c[i - 1] + b[i]
            a[i] = int(255 * c[i])
    for i in range(w):
        for j in range(h):
            img[i, j] = a[image[i, j]]
    return img
def test(model, device, train_loader,te,imgs):
        model.eval()
        total_loss = 0.
        index = 0
        img=[]
        with torch.no_grad():
            prediction = []
            file = open('prediction.txt', 'w')
            for idx, (data, target) in enumerate(train_loader):
                data = Variable(torch.unsqueeze(data, dim=1).float(), requires_grad=False)
                data, target = data.to(device), target.to(device)

                target = target.long()
                output = model(data)
                total_loss += F.nll_loss(output, target, reduction="sum").item()
                output=output.numpy()
                pre = abs(output)
                pre  = np.argmax(pre, axis=1)
                for i in range(3):
                    a=int(pre[i])
                    img.append(np.array(te[a]))
                strs = "the prediction result is"+str(pre)+", the real answer is "+str(target)+"\n"
                file.write(strs)
                prediction.append(str)
            # img = np.array(img)
            # a,b,c = img.shape
            # img = np.reshape(img,(-1,b*c))
            # plt.axis('off')
            # plt.imshow(img,plt.cm.gray)
            # plt.savefig('annotation.tif')
            originalImgX, originalImgY = Gen(imgs)
            img_equ = Equalize(originalImgY, imgs)
            plt.axis('off')
            plt.imshow(np.array(img_equ), plt.cm.gray)
            plt.savefig('annotation.tif')


lr = 0.01
momentum = 0.5
load,index,te,imgs = ReadImageAndSplitByBian(1,16)
model = Net(index)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for epoch in range(1):
    train(model, device, load, optimizer, epoch)
test(model, device, load,te,imgs)
torch.save(model.state_dict(), "fashion_mnist_cnn.pt")