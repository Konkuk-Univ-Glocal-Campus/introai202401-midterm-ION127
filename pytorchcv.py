import builtins
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import random_split
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import os
import zipfile 

# 기본 디바이스 설정
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

#조건 1-I .torchvision module에서 제공하는 FashionMNIST를 활용해야 합니다 (전체 데이터)
# FashionMNIST 데이터셋 로드
# Train 데이터와 Test 데이터셋을 가져옴
# Test 데이터를 Valid 데이터와 Test 데이터로 나눔
# dataloader를 사용하여 데이터를 로드
def load_mnist(batch_size=100):
    builtins.data_train = torchvision.datasets.FashionMNIST('./data',
        download=True,train=True,transform=ToTensor())
    builtins.data_test = torchvision.datasets.FashionMNIST('./data',
        download=True,train=False,transform=ToTensor())
    builtins.data_valid, builtins.data_test = random_split(data_test,[7000, 3000])
    data_valid.dataset.train = False

    builtins.train_loader = torch.utils.data.DataLoader(data_train,batch_size=batch_size, shuffle=True)
    builtins.valid_loader = torch.utils.data.DataLoader(data_valid, batch_size= batch_size, shuffle=True)
    builtins.test_loader = torch.utils.data.DataLoader(data_test,batch_size=batch_size, shuffle=True)

# 레이블 출력 함수
def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

# 에폭 동안의 학습 수행
def train_epoch(net,dataloader,lr=0.001,optimizer=None,loss_fn = nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    net.train() 
    total_loss,acc,count = 0,0,0
    for features,labels in dataloader:
        optimizer.zero_grad() 
        lbls = labels.to(default_device)
        out = net(features.to(default_device))
        loss = loss_fn(out,lbls) 
        loss.backward() 
        optimizer.step()
        total_loss+=loss
        _,predicted = torch.max(out,1) 
        acc+=(predicted==lbls).sum() 
        count+=len(labels)
    return total_loss.item()/count, acc.item()/count 

# 검증 수행
def validate(net, dataloader,loss_fn=nn.NLLLoss()):
    net.eval() 
    count,acc,loss = 0,0,0
    with torch.no_grad():
        for features,labels in dataloader:
            lbls = labels.to(default_device)
            out = net(features.to(default_device))
            loss += loss_fn(out,lbls) 
            pred = torch.max(out,1)[1]
            acc += (pred==lbls).sum()
            count += len(labels)
    return loss.item()/count, acc.item()/count

# 학습 수행 [model snapshot]  overfiiting을 줄이기위한 방법 중 한가지.
# 조건 2- II 모델을 컴파일하고, 적절한 손실 함수와 최적화 알고리즘을 선택합니다.
def train(net, train_loader, test_loader, optimizer=None, lr=0.001, epochs=5, loss_fn=nn.NLLLoss()):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr=lr)
    best_val_loss = float('inf')  # 초기 검증 오차를 무한대로 설정하여 어떠한 모델보다도 낮은 값으로 설정
    best_model_state = None  # 가장 낮은 검증 오차를 갖는 모델의 상태를 저장하기 위한 변수
    res = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(epochs):
        tl, ta = train_epoch(net, train_loader, optimizer=optimizer, lr=lr, loss_fn=loss_fn)
        vl, va = validate(net, test_loader, loss_fn=loss_fn)
        print(f"Epoch {ep+1:2}, Train acc={ta:.3f}, Val acc={va:.3f}, Train loss={tl:.3f}, Val loss={vl:.3f}")
        res['train_loss'].append(tl)
        res['train_acc'].append(ta)
        res['val_loss'].append(vl)
        res['val_acc'].append(va)

        # 현재 검증 오차가 이전 최적 모델보다 낮은 경우에만 모델 스냅샷 저장
        if vl < best_val_loss:
            best_val_loss = vl
            best_model_state = net.state_dict()

    # 가장 낮은 검증 오차를 갖는 모델의 상태를 사용하여 모델 재구성
    if best_model_state is not None:
        net.load_state_dict(best_model_state)

    return res

# 장기 학습 수행
def train_long(net,train_loader,test_loader,epochs=5,lr=0.01,optimizer=None,loss_fn = nn.NLLLoss(),print_freq=10):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    for epoch in range(epochs):
        net.train()
        total_loss,acc,count = 0,0,0
        for i, (features,labels) in enumerate(train_loader):
            lbls = labels.to(default_device)
            optimizer.zero_grad()
            out = net(features.to(default_device))
            loss = loss_fn(out,lbls)
            loss.backward()
            optimizer.step()
            total_loss+=loss
            _,predicted = torch.max(out,1)
            acc+=(predicted==lbls).sum()
            count+=len(labels)
            if i%print_freq==0:
                print("Epoch {}, minibatch {}: train acc = {}, train loss = {}".format(epoch,i,acc.item()/count,total_loss.item()/count))
        vl,va = validate(net,test_loader,loss_fn)
        print("Epoch {} done, validation acc = {}, validation loss = {}".format(epoch,va,vl))

# 결과 시각화
def plot_results(hist):
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(hist['train_acc'], label='Training acc')
    plt.plot(hist['test_acc'], label='Validation acc')
    plt.legend()
    plt.subplot(122)
    plt.plot(hist['train_loss'], label='Training loss')
    plt.plot(hist['test_loss'], label='Validation loss')
    plt.legend()

# 합성곱 출력 시각화
def plot_convolution(t,title='',):
    with torch.no_grad():
        c = nn.Conv2d(kernel_size=(3,3),out_channels=1,in_channels=1)
        c.weight.copy_(t)
        fig, ax = plt.subplots(2,6,figsize=(8,3))
        fig.suptitle(title,fontsize=16)
        for i in range(5):
            im = data_train[i][0]
            ax[0][i].imshow(im[0])
            ax[1][i].imshow(c(im.unsqueeze(0))[0][0])
            ax[0][i].axis('off')
            ax[1][i].axis('off')
        ax[0,5].imshow(t)
        ax[0,5].axis('off')
        ax[1,5].axis('off')
        plt.show()

# 데이터셋 시각화
def display_dataset(dataset, n=10,classes=None):
    fig,ax = plt.subplots(1,n,figsize=(15,3))
    mn = min([dataset[i][0].min() for i in range(n)])
    mx = max([dataset[i][0].max()])