import os
import random
import io

import sys
import argparse
import time
import yaml
from datetime import datetime
from torch.autograd import Variable
import matplotlib.pyplot as plt
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
# torch.autograd.set_detect_anomaly(True)
from conf import settings
from utils import get_network, get_test_dataloader, get_val_dataloader, WarmUpLR, most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, \
    update, get_mean_std, Acc_Per_Context, Acc_Per_Context_Class, penalty, cal_acc, get_custom_network,  \
    save_model, load_model, get_parameter_number, init_training_dataloader
from train_module import train_env_ours, auto_split, refine_split, update_pre_optimizer, update_pre_optimizer_vit, update_bias_optimizer
from eval_module import eval_training, eval_best, eval_mode, eval_accloss
from timm.scheduler import create_scheduler
# from draw_CAM import draw_CAM

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutout(img,n_holes,length):
    #ipdb.set_trace()
    bs = img.size(0)
    c = img.size(1)
    h = img.size(2) 
    w = img.size(3) 

    for i in range(bs):
        for j in range(c):
            
            mask = np.ones((h, w), np.float32) #

            for n in range(n_holes): 
                y = np.random.randint(h) 
                x = np.random.randint(w) 

                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0. 

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img[i][j])
            img[i][j] = img[i][j] * mask
    return img



def train(epoch):

    start = time.time()
    net.train()
    train_correct = 0.
    train_loss = []
    num_updates = epoch * len(train_loader)
    
    for batch_index, (images, labels) in enumerate(train_loader):
        if epoch <= training_opt['warm']:
            warmup_scheduler.step()
        
        if 'cutout' in training_opt and training_opt['cutout'] == True:
            images = cutout(images,1,112)
            #images, labels = map(Variable, (inputs, targets_a, targets_b)) 
        
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        if 'mixup' in training_opt and training_opt['mixup'] == True:
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, use_cuda=True)
            images, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
    
        optimizer.zero_grad()
        outputs = net(images)

        if 'mixup' in training_opt and training_opt['mixup'] == True:
            loss = mixup_criterion(loss_function, outputs, targets_a, targets_b, lam)
        else:
            loss = loss_function(outputs, labels)
        train_loss.append(loss)
        loss.backward()
        optimizer.step()

        batch_correct, train_acc = cal_acc(outputs, labels)
        train_correct += batch_correct

        num_updates += 1

        if batch_index % training_opt['print_batch'] == 0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tAcc: {:0.4f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                train_acc,
                epoch=epoch,
                trained_samples=batch_index * training_opt['batch_size'] + len(images),
                total_samples=len(train_loader.dataset)
            ))

    finish = time.time()
    train_acc_all = train_correct / len(train_loader.dataset)
    train_nll = torch.stack(train_loss).mean()
    print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))
    with io.open('results_txt/' + exp_name + '/' + '-train' + '.txt', 'a', encoding='utf-8') as file:
        file.write('epoch {}\t Train Acc: {:.4f}\t Avg_Loss: {:0.3f}\n '.format(epoch, train_acc_all, train_nll.item()))
    print('epoch {} training time consumed: {:.2f}s \t Train Acc: {:.4f}'.format(epoch, finish - start, train_acc_all))
    return train_acc_all


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', type=str, required=True, help='load the config file')
    parser.add_argument('-net', type=str, default='resnet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-multigpu', action='store_true', default=False, help='use multigpu or not')
    parser.add_argument('-name', type=str, default=None, help='experiment name')
    parser.add_argument('-debug', action='store_true', default=False, help='debug mode')
    parser.add_argument('-eval', type=str, default=None, help='the model want to eval')
    # parser.add_argument('-resume', action='store_true', default=False, help='resume training')   
    
    args = parser.parse_args()
    avg_precision = []
    avg_loss = []
    avg_precision_erm = []
    mytest_acc = []
    mytest_loss = []
    mytest_acc_erm = []

    # ============================================================================
    # LOAD CONFIGURATIONS
    with open(args.cfg) as f:
        config = yaml.safe_load(f)
    args.net = config['net']
    # args.debug = False
    training_opt = config['training_opt']
    variance_opt = config['variance_opt']
    exp_name = args.name if args.name is not None else config['exp_name']

    if 'mixup' in training_opt and training_opt['mixup'] == True:
        print('use mixup ...')
    if 'cutout' in training_opt and training_opt['cutout'] == True:
        print('use cutout ...')
    # ============================================================================
    # SEED
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(training_opt['seed'])
    if if_cuda:
        torch.cuda.manual_seed(training_opt['seed'])
        torch.cuda.manual_seed_all(training_opt['seed'])
    random.seed(training_opt['seed'])
    np.random.seed(training_opt['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ============================================================================
    # MODEL
    if variance_opt['mode'] in ['ours']:
        net = get_custom_network(args, variance_opt)
    else:
        net = get_network(args)
    
    if 'env_type' in variance_opt and variance_opt['env_type'] in ['auto-baseline', 'auto-iter'] and variance_opt['from_scratch']:
        print('load reference model ...')
        ref_arg = argparse.ArgumentParser()
        ref_arg.net = 'resnet18'
        ref_arg.gpu = args.gpu
        ref_net = get_network(ref_arg)

        load_model(ref_net, variance_opt['ref_model_path'])

        ref_net.eval()
        print('Done.')
    get_parameter_number(net)

    
    # ============================================================================
    # DATA PREPROCESSING
    if config['dataset'] is not 'Cifar':
        mean, std = training_opt['mean'], training_opt['std']
    else:
        mean, std = settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD
    train_loader_init = init_training_dataloader(config, mean, std, variance_opt['balance_factor'])

    if 'env_type' in variance_opt and variance_opt['env_type'] in ['auto-baseline', 'auto-iter', 'auto-iter-cluster']:
        pre_train_loader, pre_optimizer, pre_schedule = train_loader_init.get_pre_dataloader(batch_size=128, num_workers=4, shuffle=True, n_env=variance_opt['n_env'])
        if variance_opt['from_scratch']:
            updated_split_onehot = torch.zeros_like(pre_train_loader.dataset.soft_split).scatter_(1, torch.argmax(pre_train_loader.dataset.soft_split, 1).unsqueeze(1), 1)
            train_loader = train_loader_init.get_env_dataloader(config, training_opt['batch_size'], num_workers=4, shuffle=True, pre_split=updated_split_onehot)
            pre_split_softmax, pre_split = auto_split(ref_net, pre_train_loader, pre_optimizer, pre_schedule, pre_train_loader.dataset.soft_split, exp_name,config)
            updated_split_onehot = torch.zeros_like(pre_split_softmax).scatter_(1, torch.argmax(pre_split_softmax, 1).unsqueeze(1), 1)
            train_loader = train_loader_init.get_env_dataloader(config, training_opt['batch_size'], num_workers=4, shuffle=True, pre_split=updated_split_onehot)
            np.save('misc/banloss/NICO++'+'_'+'pre.npy', pre_split.detach().cpu().numpy())
            pre_train_loader.dataset.soft_split = pre_split
            exit()
        else:
            pre_split = np.load('misc/after_train_ours_resnet18_savenpy.npy')
            pre_split = torch.from_numpy(pre_split).cuda()
            #random init split
            pre_split = torch.randn_like(pre_split)
            pre_train_loader.dataset.soft_split = torch.nn.Parameter(pre_split)
            
            #pre_train_loader.dataset.soft_split = torch.nn.Parameter(torch.randn_like(pre_split))
            pre_split_softmax = F.softmax(pre_split, dim=-1)

        pre_split = torch.zeros_like(pre_split_softmax).scatter_(1, torch.argmax(pre_split_softmax, 1).unsqueeze(1), 1)

    else:
        pre_split = None

    if 'resnet' in args.net:
        dim_classifier = 512
        #dim_classifier = 2048
    else:
        dim_classifier = 256
    if 'env_type' in variance_opt and variance_opt['env_type'] == 'auto-iter':
        bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
        bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
        bias_dataloader = train_loader_init.get_bias_dataloader(batch_size=128, num_workers=4, shuffle=True)

    if 'env' in variance_opt:

        meta_list = [[0,1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8,9]]
        train_loader = train_loader_init.get_env_dataloader(config, training_opt['batch_size'], num_workers=4, shuffle=True, pre_split=pre_split)
            
    else:
        
        train_loader = train_loader_init.get_dataloader(training_opt['batch_size'], num_workers=4, shuffle=True)


    val_loader = get_val_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )


    test_loader = get_test_dataloader(
        config,
        mean,
        std,
        num_workers=4,
        batch_size=training_opt['batch_size'],
        shuffle=False
    )

    loss_function = nn.CrossEntropyLoss()
    
    if args.eval is not None:
        val_acc = eval_mode(config, args, net, val_loader, loss_function, args.eval)
        test_acc = eval_mode(config, args, net, test_loader, loss_function, args.eval)
        print('Val Score: %s  Test Score: %s' %(val_acc.item(), test_acc.item()))
        exit()


    if variance_opt['mode'] in ['ours']:
        assert isinstance(net, list)
        if variance_opt['sp_flag']:
            optimizer = []
            ### add classifier optimizer
            optimizer.append(optim.SGD(nn.ModuleList(net[:-1]).parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4))
            optimizer.append(optim.SGD(net[-1].parameters(), lr=training_opt['lr']*1.0, momentum=0.9, weight_decay=5e-4))
            train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer_, milestones=training_opt['milestones'], gamma=0.2) for optimizer_ in optimizer]
            iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
            warmup_scheduler = [WarmUpLR(optimizer_, iter_per_epoch * training_opt['warm']) for optimizer_ in optimizer]
        else:
            optimizer = []
            optimizer.append(optim.SGD(nn.ModuleList(net).parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4))
            train_scheduler = [optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=training_opt['milestones'], gamma=0.2)]  # learning rate decay
            iter_per_epoch = len(train_loader[0])
            warmup_scheduler = [WarmUpLR(optimizer[0], iter_per_epoch * training_opt['warm'])]

    else:
        optimizer = optim.SGD(net.parameters(), lr=training_opt['lr'], momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_opt['milestones'], gamma=0.2)  # learning rate decay
        iter_per_epoch = len(train_loader[0]) if isinstance(train_loader, list) else len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * training_opt['warm'])


    if config['resume']:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, exp_name)

    if args.debug:
        checkpoint_path = os.path.join(checkpoint_path, 'debug')

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, exp_name))


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    #create text folder
    result_path = 'results_txt/' + exp_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    best_acc = 0.
    best_vacc = 0.
    best_epoch = 0
    best_vepoch = 0
    best_train_acc = 0.
    if 'pretrain' in config and config['pretrain'] is not None:
        state_dict = torch.load(config['pretrain'])
        net.load_state_dict(state_dict, strict=False)
        print('Loaded pretrained model...')
    if config['resume']:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    with io.open('results_txt/' +  exp_name+ '/' + '-config' + '.txt', 'a', encoding='utf-8') as file:
        file.write(config['net']+'\n')
        for k,v in config['training_opt'].items():
            file.write(str(k)+' '+str(v)+'\n')
        for k,v in config['variance_opt'].items():
            file.write(str(k)+' '+str(v)+'\n')
    start = time.time()
    for epoch in range(1, training_opt['epoch']):
        if 't2tvit' in args.net and training_opt['optim']['sched']=='cosine':
            lr_scheduler.step(epoch)
        else:
            if epoch > training_opt['warm']:
                if isinstance(train_scheduler, list):
                    for train_scheduler_ in train_scheduler:
                        train_scheduler_.step()
                else:
                    train_scheduler.step()
        if config['resume']:
            if epoch <= resume_epoch:
                continue

        ### update split
        if variance_opt['env_type'] == 'auto-iter' and epoch>=variance_opt['split_renew'] and (epoch-variance_opt['split_renew'])%variance_opt['split_renew_iters']==0:  
            bias_classifier, bias_optimizer = refine_split(bias_optimizer, bias_schedule, bias_classifier, bias_dataloader, net[-1],exp_name)
            pre_optimizer, pre_schedule = update_pre_optimizer(pre_train_loader.dataset.soft_split)
            updated_split_softmax, updated_split = auto_split([net[-1], bias_classifier], variance_opt, pre_train_loader, pre_optimizer, pre_schedule, pre_train_loader.dataset.soft_split,exp_name,config)
            num_idx = []
            s_num = torch.argmax(updated_split_softmax, 1)
            s_list = s_num.cpu().numpy().tolist()
            for i in range(4):
                num_idx.append(s_list.count(i))
            #the second time
            if 'gb' in variance_opt and variance_opt['gb'] == True:
                soft_split2 = torch.nn.Parameter(torch.randn_like(pre_train_loader.dataset.soft_split))
                bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
                bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
                bias_classifier, bias_optimizer = refine_split(bias_optimizer, bias_schedule, bias_classifier, bias_dataloader, net[-1],exp_name)
                pre_optimizer, pre_schedule = update_pre_optimizer(soft_split2)
                updated_split_softmax2, updated_split2 = auto_split([net[-1], bias_classifier], variance_opt, pre_train_loader, pre_optimizer, pre_schedule, soft_split2,exp_name,config)
                s_num2 = torch.argmax(updated_split_softmax2, 1)
                s_list2 = s_num2.cpu().numpy().tolist()
                for i in range(4):
                    num_idx.append(s_list2.count(i))
                print(num_idx)

                updated_softmax = updated_split_softmax + updated_split_softmax2
                s_num3 = torch.argmax(updated_softmax, 1)
                s_list3 = s_num3.cpu().numpy().tolist()
                for i in range(4):
                    num_idx.append(s_list3.count(i))
                print(num_idx)
                updated_split_onehot = torch.zeros_like(updated_softmax).scatter_(1, torch.argmax(updated_softmax, 1).unsqueeze(1), 1)
            else:
                updated_split_onehot = torch.zeros_like(updated_split_softmax).scatter_(1, torch.argmax(updated_split_softmax, 1).unsqueeze(1), 1)
            pre_train_loader.dataset.soft_split = torch.nn.Parameter(torch.randn_like(updated_split))
            
            with io.open('results_txt/' +  exp_name+ '/' + '-addSplit' + '.txt', 'a', encoding='utf-8') as file:
                file.write(str(num_idx))
                file.write('\n')

            train_loader = train_loader_init.get_env_dataloader(config, training_opt['batch_size'], num_workers=4, shuffle=True, pre_split=updated_split_onehot)
            bias_classifier = nn.Linear(dim_classifier, training_opt['classes']).cuda()
            bias_optimizer, bias_schedule = update_bias_optimizer(bias_classifier.parameters())
            print('Update Dataloader Done')
        
        if 'env' in variance_opt:
            train_acc = train_env_ours(epoch, net, train_loader, args, training_opt, variance_opt, loss_function, optimizer, warmup_scheduler,avg_precision,avg_loss,exp_name)

        else:
            train_acc = train(epoch)
            
        t_acc,t_loss = eval_accloss(config, args, net, test_loader, loss_function, writer, epoch, exp_name)
        v_acc,v_loss = eval_accloss(config, args, net, val_loader, loss_function, writer, epoch, exp_name)
        
        with io.open('results_txt/' + exp_name+ '/' + '-test&val' + '.txt', 'a', encoding='utf-8') as file:
            file.write('epoch {}\t Test Acc: {:.4f}\t Val Acc: {:.4f}\t Test_Loss: {:0.3f}\t Test_vLoss: {:0.3f}\n'.format(epoch, t_acc, v_acc, t_loss, v_loss))        
        
        mytest_acc.append(t_acc)
        mytest_loss.append(t_loss)
        acc = t_acc
        acc2 = v_acc

        if best_acc < acc:
            if epoch >= 80:
                save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            best_epoch = epoch
            best_train_acc  = train_acc
        
        if best_vacc < acc2:
            best_vacc = acc2
            best_vepoch = epoch


        if not epoch % training_opt['save_epoch']:
            save_model(net, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        print("Train Acc: %.4f \t Best Acc: %.4f \t Best Epoch: %d" %(best_train_acc, best_acc, best_epoch))
        with io.open('results_txt/' + exp_name+ '/' + '-best_update' + '.txt', 'a', encoding='utf-8') as file:
            file.write('Best epoch {}\t Best Acc: {:.4f}\t Best vAcc: {:.4f}\t Train Acc: {:0.3f}\n'.format(epoch, best_acc, best_vacc, best_train_acc))

    print('Evaluate Best Epoch %d ...' %(best_epoch))
    acc_final = eval_best(config, args, net, test_loader, loss_function ,checkpoint_path, best_epoch)
    txt_write = open("results_txt/" + exp_name + '/-result' + '.txt', 'w')

    txt_write.write('epoch: ' + str(best_epoch) + '\n')
    txt_write.write('best_train_acc: ' + str(best_train_acc.item()) + '\n')
    txt_write.write('best_acc: ' + str(best_acc.item()) + '\n')
    txt_write.write('best vepoch: ' + str(best_vepoch) + '\n')
    txt_write.write('best_vacc: ' + str(best_vacc.item()) + '\n')
    txt_write.close
    writer.close()
    finish = time.time()
    
    with io.open('results_txt/' +  exp_name+ '/' + '-train' + '.txt', 'a', encoding='utf-8') as file:
        file.write('training time consumed: {:.2f}s '.format(finish - start))
    
    plt.switch_backend('agg')
    total_epoch = [i for i in range(epoch)]
    plt.plot(total_epoch, avg_precision, 'r.-', label="train_acc", linewidth=0.5, markersize=0)
    #plt.plot(total_epoch, avg_precision_erm, 'b.-', label="erm_acc", linewidth=0.5, markersize=0)
    plt.title('ours_resnet18_train_acc')
    #plt.legend([u'train_acc'])
    plt.legend()
    plt.xlabel('EPOCH')
    plt.savefig('results_txt/' + exp_name+ '/' + '-train_acc'+ '.png')
    plt.clf()
    
    plt.plot(total_epoch, avg_loss, 'r.-', label="train_loss", linewidth=0.5, markersize=0)
    plt.title('ours_resnet18_train_loss')
    plt.legend([u'train_loss'])
    plt.xlabel('EPOCH')
    plt.savefig('results_txt/' + exp_name + '/'+ '-train_loss'+ '.png')
    plt.clf()
    
    plt.plot(total_epoch, mytest_acc, 'r.-', label="test_acc", linewidth=0.5, markersize=0)
    #plt.plot(total_epoch, mytest_acc_erm, 'b.-', label="erm_test_acc", linewidth=0.5, markersize=0)
    plt.title('ours_resnet18_test_acc')
    #plt.legend([u'test_acc'])
    plt.legend()
    plt.xlabel('EPOCH')
    plt.savefig('results_txt/' + exp_name + '/'+ '-test_acc' + '.png')
    plt.clf()
    
    plt.plot(total_epoch, mytest_loss, 'b.-', label="test_acc", linewidth=0.5, markersize=0)
    plt.title('ours_resnet18_test_loss')
    plt.legend([u'test_loss'])
    plt.xlabel('EPOCH')
    plt.savefig('results_txt/' + exp_name + '/'+ '-test_loss' + '.png')

    
    
    
    
    
    