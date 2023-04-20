'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy

from models import *
from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net=ResNet18()
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

base_net = ResNet18()
base_net = base_net.to(device)

if device == 'cuda':
    base_net = torch.nn.DataParallel(base_net)
    cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
                         
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Load checkpoint.
print("Loading pre-trained model")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']
net.train()
train_loss = 0
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()
    acc = (correct/total)*100
    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                 % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
net.eval()
test_loss = 0
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         


def test_train():
    #print('\nEpoch: %d' % epoch)
    prune_net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = prune_net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = (correct/total)*100
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return acc
    
def test_final():
    global best_acc
    prune_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = prune_net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print("Loading single shot pruned model with sparsity of 0.5")
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/50.0percent_single_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()

print("Loading single shot pruned model with sparsity of 0.75")
print("==> Resuming from checkpoint..")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/75.0percent_single_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()


print("Loading single shot pruned model with sparsity of 0.9")
print("==> Resuming from checkpoint..")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/90.0percent_single_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()

print("Loading iteratively pruned model with sparsity of 0.5")
print("==> Resuming from checkpoint..")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/50.0percent_iter_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()

print("Loading iteratively pruned model with sparsity of 0.75")
print("==> Resuming from checkpoint..")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/75.0percent_iter_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()

print("Loading iteratively pruned model with sparsity of 0.9")
print("==> Resuming from checkpoint..")
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
prune_net = torch.load('./checkpoint/90.0percent_iter_ckpt.pth')
print("Sparsity run..")
acc = test_train()
test_final()


 


