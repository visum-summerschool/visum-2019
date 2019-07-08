#
# Perform CNN classification on CIFAR-10.
#

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torchvision import datasets, transforms

import resnet

#
# Load the CIFAR 10 dataset.
#
def load_cifar10(basedir, batch_size, kwargs):
    # Input channels normalization.
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010])

    # Load train data.
    trainloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=basedir+'cifar10/', train=True,
                    transform=transforms.Compose([
                            transforms.RandomCrop(32, 4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                    ]), download=True),
             batch_size=batch_size, shuffle=True, **kwargs)
    
    # Load test data.
    testloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=basedir+'cifar10/', train=False,
                    transform=transforms.Compose([
                            transforms.ToTensor(),
                            normalize,
                    ])),
            batch_size=batch_size, shuffle=True, **kwargs)
    
    return trainloader, testloader

################################################################################
# Training epoch.
################################################################################

#
# Main function for training.
#
def main_train(model, device, trainloader, optimizer, f_loss, epoch):
    # Set mode to training.
    model.train()
    avgloss, avglosscount = 0., 0.
    
    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        data   = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()
        
        # Compute outputs and losses.
        output = model(data)
        loss = f_loss(output, target)
        
        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss.
        avgloss += loss.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount
        
        # Print updates.
        print "Training epoch %d: loss %8.4f - %.0f\r" \
                %(epoch, newloss, 100.*(bidx+1)/len(trainloader)),
        sys.stdout.flush()
    print

################################################################################
# Testing epoch.
################################################################################

#
# Main function for testing.
#
def main_test(model, device, testloader):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    cos = nn.CosineSimilarity(eps=1e-9)
    acc = 0
    
    ty, tp = [], []
    
    # Go over all batches.
    with torch.no_grad():
        for data, target in testloader:
            # Data to device.
            data = torch.autograd.Variable(data).cuda()
            target = target.cuda(async=True)
            target = torch.autograd.Variable(target)
            
            # Forward.
            output = model(data).float()
                
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target.view_as(pred)).sum().item()
            ty.append(target.data.cpu().numpy())
            tp.append(pred.data.cpu().numpy())
    
    ty = np.concatenate(ty).astype(int)
    tp = np.concatenate(tp).astype(int)[:,0]
    acc = np.mean(ty == tp)
    
    # Print results.
    testlen = len(testloader.dataset)
    print "Testing: classification accuracy: %d/%d - %.3f" \
            %(np.sum(ty == tp), testlen, 100. * acc)

################################################################################
# Main entry point of the script.
################################################################################

#
# Parse all user arguments.
#
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 classification")
    parser.add_argument("--datadir", dest="datadir", default="data/", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=100, type=int)
    parser.add_argument("--drop2", dest="drop2", default=200, type=int)
    args = parser.parse_args()
    return args

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args     = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device   = torch.device("cuda")
    kwargs   = {'num_workers': 64, 'pin_memory': True}

    # Set the random seeds.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Load data.
    trainloader, testloader = load_cifar10(args.datadir, args.batch_size, kwargs)
    
    # Load the model.
    model = resnet.ResNet(8, 10)
    model = model.to(device)
    
    # Load the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, \
            momentum=args.momentum, weight_decay=args.decay)
    
    # Initialize the loss functions.
    f_loss = nn.CrossEntropyLoss().cuda()

    # Main loop.
    learning_rate = args.learning_rate
    for i in xrange(args.epochs):
        #print "---"
        # Learning rate decay.
        if i in [args.drop1, args.drop2]:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        # Train and test.
        main_train(model, device, trainloader, optimizer, f_loss, i)
        main_test(model, device, testloader)
