#/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device.type=='cuda':
    print(torch.cuda.get_device_name(0))


CLASSES=10
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1, 96, 11, stride=4, padding=0)
        self.relu1 = nn.ReLU()

        # Layer 2
        self.conv2 = nn.Conv2d(96, 96, 1)
        self.relu2 = nn.ReLU()

        # Layer 3
        self.conv3 = nn.Conv2d(96, 96, 1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.drop3 = nn.Dropout(0.5)

        # Layer 4
        self.conv4 = nn.Conv2d(96, 256, 11, stride=4, padding=2)
        self.relu4 = nn.ReLU()

        # Layer 5 (first occurrence, assuming typo in layer naming)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.relu5 = nn.ReLU()

        # Layer 5 (second occurrence, corrected to Layer 6 for clarity)
        self.conv6 = nn.Conv2d(256, 256, 1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(3, stride=2)
        self.drop6 = nn.Dropout(0.5)

        # Layer 7
        self.conv7 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.relu7 = nn.ReLU()

        # Layer 8
        self.conv8 = nn.Conv2d(384, 384, 1)
        self.relu8 = nn.ReLU()

        # Layer 9
        self.conv9 = nn.Conv2d(384, 384, 1)
        self.relu9 = nn.ReLU()
        self.drop9 = nn.Dropout(0.5)

        # Layer 10
        self.conv10 = nn.Conv2d(384, 10, 3, stride=1, padding=1)
        self.relu10 = nn.ReLU()

        # Layer 11
        self.conv11 = nn.Conv2d(10, 10, 1)
        self.relu11 = nn.ReLU()

        # Layer 12
        self.conv12 = nn.Conv2d(10, 10, 1)
        self.relu12 = nn.ReLU()
        self.adapool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.drop3(self.pool3(self.relu3(self.conv3(x))))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.drop6(self.pool6(self.relu6(self.conv6(x))))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        x = self.drop9(self.relu9(self.conv9(x)))
        x = self.relu10(self.conv10(x))
        x = self.relu11(self.conv11(x))
        x = self.adapool(self.relu12(self.conv12(x)))
        x = torch.flatten(x, 1)  # Flatten for potential further layers or a classifier
        return x

transform_conf=transforms.Compose([
    transforms.Resize((227,227)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])


BATCH_SIZE=16
train_dataset=datasets.MNIST('/users/hutruon/data',train=True,download=True,transform=transform_conf,)
test_dataset=datasets.MNIST('/users/hutruon/data/',train=False,download=True,transform=transform_conf)



train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)


model=Net().to(device)
optimizer=optim.Adam(params=model.parameters(),lr=0.0001)
loss_fn = nn.CrossEntropyLoss()


def train(model,device,train_loader,optimizer,epochs):
    print("inside train")
    model.train()
    for batch_ids, (img, classes) in enumerate(train_loader):
        classes=classes.type(torch.LongTensor)
        img,classes=img.to(device),classes.to(device)
        torch.autograd.set_detect_anomaly(True)     
        optimizer.zero_grad()
        output=model(img)
        loss = loss_fn(output,classes)                
        
        loss.backward()
        optimizer.step()
    if(batch_ids +1) % 2 == 0:
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_ids* len(img), len(train_loader.dataset),
            100.*batch_ids / len(train_loader),loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for img,classes in test_loader:
            img,classes=img.to(device), classes.to(device)
            y_hat=model(img)
            test_loss+=F.nll_loss(y_hat,classes,reduction='sum').item()
            _,y_pred=torch.max(y_hat,1)
            correct+=(y_pred==classes).sum().item()
        test_loss/=len(test_dataset)
        print("\n Test set: Avarage loss: {:.0f},Accuracy:{}/{} ({:.0f}%)\n".format(
            test_loss,correct,len(test_dataset),100.*correct/len(test_dataset)))
        print('='*30)


if __name__=='__main__':
    seed=42
    EPOCHS=2
    
    for epoch in range(1,EPOCHS+1):
        train(model,device,train_loader,optimizer,epoch)
        test(model,device,test_loader)




