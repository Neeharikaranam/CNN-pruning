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
import copy

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('-pr',type=float)
parser.add_argument('-shot', type=str)

args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
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

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

net=ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def prune_train():
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
                     
def prune_test():
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
    acc = correct/total
    return acc

def test(epoch):
    global best_acc
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
        
if (args.pr == None):  
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()
else:
    print('==> Building model..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    prune_net = copy.deepcopy(net)
    params_to_prune =(
        (prune_net.module.conv1, 'weight'),
        (prune_net.module.layer1[0].conv1,'weight'),
        (prune_net.module.layer1[0].conv2,'weight'),
        (prune_net.module.layer1[1].conv1,'weight'),
        (prune_net.module.layer1[1].conv2,'weight'),
        (prune_net.module.layer2[0].conv1,'weight'),
        (prune_net.module.layer2[0].conv2,'weight'),
        (prune_net.module.layer2[1].conv1,'weight'),
        (prune_net.module.layer2[1].conv2,'weight'),
        (prune_net.module.layer2[0].shortcut[0],'weight'),
        (prune_net.module.layer3[0].conv1,'weight'),
        (prune_net.module.layer3[0].conv2,'weight'),
        (prune_net.module.layer3[0].shortcut[0],'weight'),
        (prune_net.module.layer3[1].conv1,'weight'),
        (prune_net.module.layer3[1].conv2,'weight'),
        (prune_net.module.layer4[0].conv1,'weight'),
        (prune_net.module.layer4[0].conv2,'weight'),
        (prune_net.module.layer4[0].shortcut[0],'weight'),
        (prune_net.module.layer4[1].conv1,'weight'),
        (prune_net.module.layer4[1].conv2,'weight'),
        (prune_net.module.linear,'weight'),

    )
    if (args.shot == 'single'):
        print("Singel shot Sparsity run..")
        prune.global_unstructured(
            params_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=args.pr,
        )

        zero_sum=0
        total_sum=0
        list_all_elements=[]
        for i in params_to_prune:
            zero_sum+=torch.sum(i[0].weight==0)
    
        for i in params_to_prune:
            total_sum+=i[0].weight.nelement()
     
        print(
            "Global sparsity: {:.2f}%".format(
            100. * float(zero_sum)/float(total_sum)
          
        ))
    else:
        # In iterative pruning we train the model until we get the specified global sparsity 
        if (args.shot == 'iter'):
            print("Iterative Shot Sparsity run..")
            sparsity = args.pr * 100
            iter_sparsity = 0.11
            global_sparsity = 0
            num_iter = 10
            for i in range(100):
                if (global_sparsity > sparsity):
                    print("hurray!")
                    break
                prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=iter_sparsity,
                )
                zero_sum=0
                total_sum=0
                list_all_elements=[]
                for param in params_to_prune:
                    zero_sum+=torch.sum(param[0].weight==0)
    
                for param in params_to_prune:
                    total_sum+=param[0].weight.nelement()
     
                global_sparsity = float(zero_sum)/float(total_sum)*100
                print(
                    "Global sparsity: {:.2f}%".format(
                    100 * (zero_sum/total_sum)))
                
                for j in range(5):
                    k= str(j+1)
                    l = str(i+1)
                    print("Epoch " + k + " in iteration " + l )
                    prune_train()
        else:
            print("Wrong arguments")

    for epoch in range(start_epoch, start_epoch+20):
        prune_train()
        prune_test()
        scheduler.step()
    
    for _,module in prune_net.named_modules():
        try:
            prune.remove(module, "weight")
        except:
            pass 
    
    print('Saving Prune Model..')
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if (args.shot == 'iter'):
        path = './checkpoint/' + str(args.pr*100) + 'percent_' + 'iter_ckpt.pth'
    else:
        path = './checkpoint/' + str(args.pr*100) + 'percent_' + 'single_ckpt.pth'
    torch.save(prune_net, path)


