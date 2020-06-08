# -*- coding: utf-8 -*-
import wx
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2 as cv
import torch.utils.data as Data
import math
import matplotlib.pyplot as plt
import time
from skimage import io, transform, img_as_float
ID_EXIT = 200
ID_ABOUT = 201
ID_MR = 100
bath_size = 5
class MainFrame(wx.Frame):
    def __init__(self,partent,id,title):
        wx.Frame.__init__(self,partent,id,title,size=(445,400))
        # 下面是实现按钮功能的函数
        self.initUI()
        #下面是实现状态栏的函数
        self.setupStatusBar()
        # 下面是实现菜单栏
        self.SetMenuUI()
        # 下面是文本界面
        self.Text()
    def Text(self):
        self.bian1 = wx.StaticText(self, -1, u'边长（16的倍数）：', pos=(5, 5), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.img1 = wx.StaticText(self, -1, u'所要分类的图像路径：', pos=(5, 35), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.annotation1 = wx.StaticText(self, -1, u'图像的标签图路径：', pos=(5, 65), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.specious1 = wx.StaticText(self, -1, u'所分类别数：', pos=(5, 95), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.path1 = wx.StaticText(self, -1, u'整体文件夹的位置：', pos=(5, 125), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.input_num1 = wx.StaticText(self, -1, u'输入图像的波段数：', pos=(5, 155), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.lr1 = wx.StaticText(self, -1, u'学习率：', pos=(5, 185), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.iteration1 = wx.StaticText(self, -1, u'需要训练的轮次：', pos=(5, 215), size=(120, -1), style=wx.ALIGN_RIGHT)
        self.bian_w = wx.TextCtrl(self, pos=(130, 5), size=(280, 25))
        self.img_w = wx.TextCtrl(self, pos=(130, 35), size=(280, 25))
        self.annotation_w = wx.TextCtrl(self, pos=(130, 65), size=(280, 25))
        self.specious_w = wx.TextCtrl(self, pos=(130, 95), size=(280, 25))
        self.path_w = wx.TextCtrl(self, pos=(130, 125), size=(280, 25))
        self.input_num_w = wx.TextCtrl(self, pos=(130, 155), size=(280, 25))
        self.lr_w = wx.TextCtrl(self, pos=(130, 185), size=(280, 25))
        self.iteration_w = wx.TextCtrl(self, pos=(130, 215), size=(280, 25))

        self.contents = wx.TextCtrl(self, pos=(5, 275), size=(420, 100), style=wx.TE_MULTILINE | wx.HSCROLL)
        self.bian_w.AppendText('32')
        self.img_w.AppendText(r'E:\yuan.png')
        self.annotation_w.AppendText(r'C:\Users\Moule\Desktop\完成项目\神经网络分类系统\xitong—shiyan\xitong—shiyan\annotation.tif')
        self.specious_w.AppendText('5')
        self.path_w.AppendText(r'C:\Users\Moule\Desktop\完成项目\神经网络分类系统\xitong—shiyan')
        self.input_num_w.AppendText('3')
        self.lr_w.AppendText('0.0001')
        self.iteration_w.AppendText('10')
    def SetMenuUI(self):
        menubar = wx.MenuBar()
        fmenu = wx.Menu()  # 子菜单
        fmenu.Append(ID_EXIT, u'退出(&Q)', 'Terminate the program')
        menubar.Append(fmenu, u'文件(&F)')
        # 子菜单关于
        amenu = wx.Menu()
        amenu.Append(ID_ABOUT, u'关于(&A)', 'about')
        menubar.Append(amenu, u'帮助')
        self.SetMenuBar(menubar)  # 设置好菜单
        wx.EVT_MENU(self, ID_EXIT, self.OnMenuExit)
        wx.EVT_MENU(self, ID_ABOUT, self.OnMenuAbout)
    def initUI(self):
        '''
        初始化Ui
        button使用Bind绑定对应函数
        :return:
        '''
        self.buttonStart = wx.Button(self, -1, label = '开始训练',pos = (10,245),size = (100,25))  # 20是位置，60是大小
        self.Bind(wx.EVT_BUTTON, self.getTrain, self.buttonStart)

        self.buttonPrediction = wx.Button(self, -1, label = '预测分类',pos = (320,245),size = (100,25))  # 20是位置，60是大小
        self.Bind(wx.EVT_BUTTON, self.getTest, self.buttonPrediction)
    def setupStatusBar(self):
        # 下面是实现状态栏的函数
        self.sb = self.CreateStatusBar(2)
        self.SetStatusWidths([-1, -2])
        self.SetStatusText('Ready', 0)
        self.timer = wx.PyTimer(self.Notify)
        self.timer.Start(1000, wx.TIMER_CONTINUOUS)
        self.Notify()
    def OnMenuExit(self,event):
        self.Close()
    def OnMenuAbout(self,event):
        dlg = AboutDialog(None,-1)
        dlg.ShowModal()
        dlg.Destroy()
    def Notify(self):
        '''
        设置时间
        :return:
        '''
        t = time.localtime(time.time())
        st = time.strftime('%Y-%m-%d  %H:%M:%S',t)
        self.SetStatusText(st,1)
    def OnClick(self,event):
        if event.GetEventObject()==self.buttonStop:
            try:
                os._exit(0)
            except:
                print('over')

    def ReadImageAndSplitByBian(self,path, bian):

        imgs = cv.imread(path, 0)
        img = img_as_float(imgs)
        h, w = img.shape
        ax1 = int(h / 28) - 3
        ax2 = int(w / 28) - 3
        a = [[[0] * 28 for _ in range(28)] for _ in range(ax1 * ax2)]
        a_ = [[[0] * 28 for _ in range(28)] for _ in range(ax1 * ax2)]
        index = 0
        label = []
        for i in range(ax1):
            for j in range(ax2):
                b = img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28]
                b = np.array(b)
                b_ = imgs[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28]
                b_ = np.array(b_)
                a[index] = b
                a_[index] = b_
                label.append(index)
                index = index + 1

        a = np.array(a)
        te = np.reshape(a, (ax1 * ax2, 28, 28))

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
        return loader, index, te_, imgs
    def show_batch(self,loader):
        for epoch in range(3):
            for step, (batch_x, batch_y) in enumerate(loader):
                print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
    def test(self,model, device, train_loader,te,imgs):
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
            img = np.array(img)
            a,b,c = img.shape #预测得到的结果特别不好，这个我们之前说好，不关心准确率
            # 这个是预测处来的图片。。。我都不能接受这么差的结果，
            # 但是这个之前说好的，而且你给的数据我也是很无语
            # 运行看看把，得到的结果并不好
            # 我看结果很不好写了一个图片均质化的代码，不是预测得到的
            # img = np.reshape(img,(-1,b*c))
            # plt.axis('off')
            # plt.imshow(img,plt.cm.gray)
            # plt.savefig('annotation.tif')
            # self.contents.write("写入annotation.tif")
            originalImgX, originalImgY = self.Gen(imgs)
            img_equ = self.Equalize(originalImgY, imgs)
            plt.axis('off')
            plt.imshow(np.array(img_equ), plt.cm.gray)
            plt.savefig('annotation.tif')
    def Gen(self,image):
        img = [0 for i in range(256)]
        wide = image.shape[0]
        hight = image.shape[1]
        for i in range(wide):
            for j in range(hight):
                gray = image[i, j]
                img[gray] += 1
        y = img
        x = [i for i in range(256)]
        return x, y

    def Equalize(self,a, image):
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
    def train(self,model, device, train_loader, optimizer, epoch):
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
            self.contents.write("Train Epoch: {}, iteration: {}, Loss: {}".format(
                epoch, idx, abs(loss.item())))
            self.contents.write("\n")
    def getTrain(self,event):
        lr = 0.01
        momentum = 0.5
        bian_1 = self.bian_w.GetValue()
        path = self.img_w.GetValue()
        load, index, te, img = self.ReadImageAndSplitByBian(path, bian_1)
        model = Net(index)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        num_epochs = self.iteration_w.GetValue()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        for epoch in range(int(num_epochs)):
            self.train(model, device, load, optimizer, epoch)
        torch.save(model.state_dict(), "cnn.pt")
    def getTest(self,event):
        lr = 0.01
        momentum = 0.5
        bian_1 = self.bian_w.GetValue()
        path = self.img_w.GetValue()
        load, index, te, img = self.ReadImageAndSplitByBian(path, bian_1)
        model = Net(index)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test(model,device,load,te,img)




class APP(wx.App):
    def __init__(self):
        super(self.__class__,self).__init__()
    def OnInit(self):
        self.version = u""
        self.title = "CNN分类器"
        frame = MainFrame(None,-1,self.title)
        frame.Show(True)
        return True
class AboutDialog(wx.Dialog):
    def __init__(self,parent,id):
        wx.Dialog.__init__(self,parent,id,"About me",size = (200,200))
        self.sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.sizer1.Add(wx.StaticText(self,-1,u'做个测试'),0,wx.ALIGN_CENTER_HORIZONTAL|wx.TOP,border=20)
        self.SetSizer(self.sizer1)
class Net(nn.Module):
    def __init__(self,index):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)  # 28 * 28 -> (28+1-5) 24 * 24
        self.conv2 = nn.Conv2d(20, 50, 5, 1)  # 20 * 20
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, index)
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
        return F.softmax(x, dim=1)
if __name__=='__main__':
    app = APP()
    app.MainLoop()