# -pytorch-MNIST-
mnist集的训练模型，准确率97%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

train_dataset=torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(), download=True)

#构建网络
class FCnet(nn.Module):
    """A Neural Network with a hidden layer"""
    def __init__(self, input_size, hidden_size, output_size):
        super(FCnet,self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.layer1(x))
        x = self.layer2(x) 
        return x

#设置参数
input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 20
batch_size = 16
learning_rate = 1e-3

model = FCnet(input_size,hidden_size,output_size)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True) 
test_loader =torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=False)

total_step = len(train_loader)

lossFunction = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

#显示数据
imgs, labels = next(iter(train_loader))
img = imgs[0,:,:,:].numpy().squeeze(0)
print('showing digit {}'.format(labels.numpy()[0]))
plt.imshow(img)
plt.show()

#训练模型
for epoch in range(num_epochs):
    for i, (img,labels) in enumerate(train_loader):
        img = img.view(img.size(0),-1)
        
        out = model(img)
        loss = lossFunction(out,labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (i+1) % 100 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#test accuracy
with torch.no_grad():
    correct = 0
    total = 0
    
    for img,labels in test_loader:
        
        img = img.view(img.size(0),-1)
        out = model(img)
        _,predicted = torch.max(out.data,1) 
        correct += (predicted == labels.data).sum() 
        total += img.data.size()[0]
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
