import argparse
import subprocess
import sys
import time
import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ffn_cifar
import cnn_cifar
import ffn as ffn_mnist
import cnn as cnn_mnist
from torch.autograd import Variable
import csv
import plot
import math


# some helper functions
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# the summed quadratic length of all module parameters
def vlen(layer):
    vsum = 0.
    vnn = 0
    for vv in layer.parameters():
        # if vv.requires_grad:
        param = vv.data
        vsum = vsum + (param * param).sum()
        vnn = vnn + param.numel()
    return vsum / vnn


def test_accuracy(net, images, labels):
    # Test the model
    net.eval()
    with torch.no_grad():
            output, _ = net(images)
            _, predicted = torch.max(output.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().to('cpu').numpy()

    net.train()
    return 100 * (correct / total)


def calculate_confidence(net, loader, dataset, approach, batchsize, title):
    net.eval()
    file = open('logslong/calibration' + str(dataset) + '-' + str(approach) + str(batchsize), 'w')
    writer = csv.writer(file)
    confidence_array = np.zeros(0)
    label_array = np.zeros(0)
    prediction_array = np.zeros(0)

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            output, _ = net(images)
            _, predicted = torch.max(output.data, 1)
            for i in range(labels.size(0)):
                confidence = np.amax(output.to('cpu').detach().numpy()[i])
                label = labels.to('cpu').numpy()[i]
                prediction = predicted.to('cpu').numpy()[i]
                confidence_array = np.append(confidence_array, confidence)
                label_array = np.append(label_array, label)
                prediction_array = np.append(prediction_array, prediction)
                writer.writerow(str(prediction) + ", " + str(label) + ", " + str(confidence))

    file.close()
    net.train()
    plot.create_diagrams(label_array, prediction_array, confidence_array, title)


def out_of_distribution(net, loader1, batchsize, title):
    if args.dataset == "cifar10":
        dataset2 = "svhn"
        trainloader2, loader2, _ = load_dataloader_and_model("svhn")
    elif args.dataset == "svhn":
        dataset2 = "cifar10"
        trainloader2, loader2, _ = load_dataloader_and_model("cifar10")
    else:
        return

    net.eval()
    with torch.no_grad():
        dataset1_x_likelihoods = np.zeros(0)
        dataset2_x_likelihoods = np.zeros(0)
        for data in loader1:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            output, nosoftout = net(images)
            for j in range(labels.size(0)):
                x_likelihood = 0
                logits = torch.exp(nosoftout.data[j]).to('cpu').numpy()
                for i in logits:
                    x_likelihood = x_likelihood + i
                x_likelihood = math.log(x_likelihood)
                if np.isfinite(x_likelihood):
                    dataset1_x_likelihoods = np.append(dataset1_x_likelihoods, x_likelihood)
        for data in loader2:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            output, nosoftout = net(images)
            for j in range(labels.size(0)):
                x_likelihood = 0
                logits = torch.exp(nosoftout.data[j]).to('cpu').numpy()
                for i in logits:
                    x_likelihood = x_likelihood + i
                x_likelihood = math.log(x_likelihood)
                if np.isfinite(x_likelihood):
                    dataset2_x_likelihoods = np.append(dataset2_x_likelihoods, x_likelihood)
                    
                    
        dataset1_x_likelihoods = np.sort(dataset1_x_likelihoods)
        dataset2_x_likelihoods = np.sort(dataset2_x_likelihoods) 
        tpr = np.zeros(0)
        fpr = np.zeros(0)
        lastfp = 0
        area = 0
        for t in dataset1_x_likelihoods: 
            tp = dataset1_x_likelihoods[dataset1_x_likelihoods>t].size
            tp = tp / dataset1_x_likelihoods.size
            tpr = np.append(tpr,tp)
            fp = dataset2_x_likelihoods[dataset2_x_likelihoods>t].size
            fp = fp / dataset2_x_likelihoods.size
            fpr = np.append(fpr,fp)
            area = area+(tp*(fp-lastfp))
            lastfp = fp
        plot.draw_roc(tpr,fpr, args.dataset, dataset2, batchsize, title,1 - area)

    plot.create_histogramm(dataset1_x_likelihoods, dataset2_x_likelihoods, args.dataset, dataset2, batchsize, title, "log p(x)")
    net.train()
    
    
def out_of_distribution_max_conditional(net, loader1, batchsize, title):
    if args.dataset == "cifar10":
        dataset2 = "svhn"
        trainloader2, loader2, _ = load_dataloader_and_model("svhn")
    elif args.dataset == "svhn":
        dataset2 = "cifar10"
        trainloader2, loader2, _ = load_dataloader_and_model("cifar10")
    else:
        return

    net.eval()
    with torch.no_grad():
        dataset1_max_conds = np.zeros(0)
        dataset2_max_conds = np.zeros(0)
        for data in loader1:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            output, nosoftout = net(images)
            for j in range(labels.size(0)):
                x_likelihood = 0
                max_con = np.amax(output.data[j].to('cpu').numpy())
                dataset1_max_conds = np.append(dataset1_max_conds, max_con)
        for data in loader2:
            images, labels = data
            images, labels = Variable(images).cuda(), labels.cuda()
            output, nosoftout = net(images)
            for j in range(labels.size(0)):
                x_likelihood = 0
                max_con = np.amax(output.data[j].to('cpu').numpy())
                dataset2_max_conds = np.append(dataset2_max_conds, max_con)
                
        dataset2_max_conds = np.sort(dataset2_max_conds)
        dataset1_max_conds = np.sort(dataset1_max_conds)
        tpr = np.zeros(0)
        fpr = np.zeros(0)
        lastfp = 0
        area = 0
        for t in dataset1_max_conds: 
            tp = dataset1_max_conds[dataset1_max_conds>t].size
            tp = tp / dataset1_max_conds.size
            tpr = np.append(tpr,tp)
            fp = dataset2_max_conds[dataset2_max_conds>t].size
            fp = fp / dataset2_max_conds.size
            fpr = np.append(fpr,fp)
            area = area+(tp*(fp-lastfp))
            lastfp = fp
        plot.draw_roc(tpr,fpr, args.dataset, dataset2, batchsize, title, 1 - area)      
  
        plot.create_histogramm(dataset1_max_conds, dataset2_max_conds, args.dataset, dataset2, batchsize, title, "max p(y|x)")
    net.train()


def load_dataloader_and_model(model):
    # case mnist - dataset
    if model == 'mnist':

        # load data
        transform_train = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,), (0.3081,))
        ])

        dataset_train = torchvision.datasets.MNIST('./', train=True, download=True, transform=transform_train)

        dataset_val = torchvision.datasets.MNIST('./', train=False, download=True, transform=transform_test)

        trainset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=2)

        testset_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.bs, shuffle=True, num_workers=2)

        # model
        if args.architecture.lower() == 'fc':
            ffn = ffn_mnist.FFN()
        elif args.architecture.lower() == 'cnn':
            ffn = cnn_mnist.FFN()

    # case cifar10 - dataset
    elif model == 'cifar10':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        dataset_train = torchvision.datasets.CIFAR10('./', train=True, download=True, transform=transform_train)

        dataset_val = torchvision.datasets.CIFAR10('./', train=False, download=True, transform=transform_test)

        trainset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=2)

        testset_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=2)

        # model
        if args.architecture.lower() == 'fc':
            ffn = ffn_cifar.FFN()

        elif args.architecture.lower() == 'cnn':
            ffn = cnn_cifar.FFN()


    # case fmnist - dataset
    elif model == 'fmnist':
        dataset_train = torchvision.datasets.FashionMNIST('./', train=True, download=True,
                                                          transform=transforms.Compose([transforms.ToTensor()]))

        dataset_val = torchvision.datasets.FashionMNIST('./', train=False, download=True,
                                                        transform=transforms.Compose([transforms.ToTensor()]))

        trainset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=2)

        testset_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=2)

        # model
        if args.architecture.lower() == 'fc':
            ffn = ffn_mnist.FFN()
        elif args.architecture.lower() == 'cnn':
            ffn = cnn_mnist.FFN()

    # case svhn - dataset
    elif model == 'svhn':

        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))])

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))])

        dataset_train = torchvision.datasets.SVHN('./', split='train', download=True, transform=transform_train)

        dataset_val = torchvision.datasets.SVHN('./', split='test', download=True, transform=transform_test)

        trainset_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True, num_workers=2)

        testset_loader = torch.utils.data.DataLoader(dataset_val, batch_size=args.bs, shuffle=False, num_workers=2)

        # model
        if args.architecture.lower() == 'fc':
            ffn = ffn_cifar.FFN()
        elif args.architecture.lower() == 'cnn':
            ffn = cnn_cifar.FFN()
    return trainset_loader, testset_loader, ffn


# entry point, the main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of the mnist/ cifar10/ fmnist/ svhn.')
    parser.add_argument('--data_dir', default='.', help='Data directory.')
    parser.add_argument('--call_prefix', default='tmp', help='Call prefix.')
    parser.add_argument('--stepsize', type=float, default=0.0001, help='Gradient step size.')
    parser.add_argument('--bs', type=int, default=256, help='Batch size')
    parser.add_argument('--iterations', type=int, default=200000, help='')
    parser.add_argument('--dataset', default='mnist', help='Database.')
    parser.add_argument('--architecture', default='cnn', help='network architecture (fc/cnn)')
    parser.add_argument('--variant', default='standard', help='variant of the Loss (standard, generative)')

    args = parser.parse_args()

    ensure_dir(args.data_dir + '/logslong')
    ensure_dir(args.data_dir + '/modelslong')

    time0 = time.time()

    titel = args.data_dir + '/logslong/' + args.dataset + '-' + args.call_prefix
    logname = titel + '.txt'
    # the first print with 'w', i.e. the file is overwritten if it exists
    print('# Starting at ' + time.strftime('%c'), file=open(logname, 'w'), flush=True)

    device = torch.cuda.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    trainset_loader, testset_loader, ffn = load_dataloader_and_model(args.dataset)
  
  
  
    #stepsize =   args.stepsize * math.sqrt(args.bs/ 4)
    #scale the learning rate compared to smallest batch size 4 learning rate like in "Train longer, generalize better: closing 
    #the generalization gap in large batch training of neural networks
    ffn.cuda()
    stepsize = args.stepsize
    print('# Data loaded,  ' + str(len(trainset_loader) * args.bs) + '/' + str(
        len(testset_loader) * args.bs) + ' samples used.', file=open(logname, 'a'), flush=True)

    log_period = len(trainset_loader)
    save_period = len(trainset_loader)*10
    conf_period = len(trainset_loader)*5
    count = 0
    
    best_test_acc = 0
    tolerance = 0
    
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(ffn.parameters(), lr=stepsize)

    print('# Go ...', file=open(logname, 'a'), flush=True)
    breakloop = False
    test_acc = 0 
    train_acc = 0
    while True:
        ffn.train()
        for i, data in enumerate(trainset_loader):
            input_data, labels = data
            input_data, labels = Variable(input_data).cuda(), Variable(labels).cuda()
            output, logit = ffn(input_data)
            optimizer.zero_grad()
            if args.variant == 'standard':
                loss = criterion(logit, labels)
                loss.backward()


            elif args.variant == 'generative':
                loss1 = (logit * torch.nn.functional.one_hot(labels, num_classes=10)).sum(-1).mean()
                x_scores = torch.log(torch.exp(logit).sum(-1))
                min_score = x_scores.min()
                x_probs = torch.softmax(x_scores - min_score, -1).detach()
                loss2 = (x_probs * x_scores).sum()
                loss = loss2 - loss1
                loss.backward()

            optimizer.step()
            
            ffn.eval()
            
            
            
            train_acc = 0.99*train_acc + 0.01*test_accuracy(ffn, input_data, labels)
            
            
            # once per epoch print something out
            if (count % log_period == log_period - 1) or (count == args.iterations - 1):
                for i, data in enumerate(testset_loader): 
                    inputs, testlabels = data
                    inputs, testlabels = Variable(inputs).cuda(), Variable(testlabels).cuda()
                    test_acc = 0.9*test_acc + 0.1*test_accuracy(ffn, inputs, testlabels)
                message = ' iteration: ' + str(count) + ' Training Accuracy: ' + str(train_acc) + ' Validation Accuracy: ' + str(test_acc)
                print(message, file=open(logname, 'a'), flush=True)
            
            
            #    if test_acc <= best_test_acc: 
            #        tolerance = tolerance + 1
            #        if tolerance >= 5: 
            #            for i, data in enumerate(testset_loader): 
            #                inputs, testlabels = data
            #                inputs, testlabels = Variable(inputs).cuda(), Variable(testlabels).cuda()
            #                test_acc = 0.9*test_acc + 0.1*test_accuracy(ffn, inputs, testlabels)
            #            message = ' iteration: ' + str(count) + ' Training Accuracy: ' + str(train_acc) + ' Validation Accuracy: ' + str(test_acc)
            #            print(message, file=open(logname, 'a'), flush=True)
            #            calculate_confidence(ffn, testset_loader, args.dataset, args.variant, args.bs, titel)
            #            out_of_distribution(ffn, testset_loader, args.bs, titel)
            #            out_of_distribution_max_conditional(ffn, testset_loader, args.bs, titel + "maxcond" )
            #            print('# Saving models ...', file=open(logname, 'a'), flush=True)
            #            torch.save(ffn.state_dict(), args.data_dir + '/modelslong/ffn-' + args.call_prefix + '.pt')
            #            print('# ... done.', file=open(logname, 'a'), flush=True)
            #            breakloop = True
            #            break
            #    elif test_acc > best_test_acc: 
            #        tolerance = 0
            #        best_test_acc = test_acc
            # once awhile calculate confidence
            if (count % conf_period == conf_period - 1) or (count == args.iterations - 1):
                calculate_confidence(ffn, testset_loader, args.dataset, args.variant, args.bs, titel)
                out_of_distribution(ffn, testset_loader, args.bs, titel)
                out_of_distribution_max_conditional(ffn, testset_loader, args.bs, titel + "maxcond" )

            # once awhile save the models for further use, save images to visualize
            if (count % save_period == save_period - 1) or (count == args.iterations - 1):
                print('# Saving models ...', file=open(logname, 'a'), flush=True)
                torch.save(ffn.state_dict(), args.data_dir + '/modelslong/ffn-' + args.call_prefix + '.pt')
                print('# ... done.', file=open(logname, 'a'), flush=True)

            count += 1
            if count == args.iterations:
                breakloop = True
                break
        if breakloop: 
            break
print('# Finished at ' + time.strftime('%c') + ', %g seconds elapsed' % (time.time() - time0),
      file=open(logname, 'a'), flush=True)
