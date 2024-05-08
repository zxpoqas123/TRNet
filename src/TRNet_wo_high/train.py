import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import soundfile as sound
import datetime
import sys, subprocess
import math
import random
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import logging
import time
from data import Mydataset
from model import FocalLoss, Mymodel, MSE
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(description="SER-Baseline")
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--seed',default=2021,type=int)
parser.add_argument('--max_len',default=6, type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument("--root",type=str,required=True)
parser.add_argument("--feature",type=str,required=True)
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument("--loss_type",default='Focal',type=str)
parser.add_argument("--optimizer_type",default='SGD',type=str)
parser.add_argument("--device_number",default='0',type=str)
parser.add_argument("--backbone_type",default='CNN6',type=str)
parser.add_argument('--snr',default=100,type=int)
parser.add_argument('--noise_type',default='ESC50',type=str)
parser.add_argument('--fold',type=int,required=True)
parser.add_argument('--do_se', action='store_true')
parser.add_argument('--alpha',default=1.0,type=float)
parser.add_argument('--beta',default=1.0,type=float)
args = parser.parse_args() 

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(model, device, train_loader, criterion_mse, criterion_task, optimizer, epoch, logger):
    model.train()
    logger.info('start training')
    lr = optimizer.param_groups[0]["lr"]
    logger.info('lr: {:.5f}'.format(lr))
    correct = 0
    for batch, data in tqdm(enumerate(train_loader)):
        spec_n, spec_e, spec, clean_ebd, emo_label = data
        spec_n, spec_e, spec, clean_ebd, emo_label = spec_n.to(device), spec_e.to(device), spec.to(device), clean_ebd.to(device), emo_label.to(device)
        optimizer.zero_grad()
        emo_output, noise_ebd, spec_new, _, proj_ebd = model(spec_n,spec_e)
        loss1 = criterion_task(emo_output, emo_label)
        loss2 = criterion_mse(spec_new, spec)
        loss3 = criterion_mse(proj_ebd, clean_ebd)
        loss = loss1 + args.alpha*loss2 + args.beta*loss3
        loss.backward()  
        nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], max_norm=10, norm_type=2)
        optimizer.step()

        pred = emo_output.argmax(dim=1, keepdim=True)
        #correct += lam * pred.eq(label_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(label_b.view_as(pred)).sum().item()
        correct += pred.eq(emo_label.view_as(pred)).sum().item()
        if batch % 20 == 0:
            logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t loss_task={:.5f}\t loss_mse1={:.5f}\t loss_mse2={:.5f}\t'.format(epoch , batch * len(emo_label), len(train_loader.dataset), 100. * batch / len(train_loader), loss1.item(), loss2.item(), loss3.item()))
    logger.info('Train set Accuracy: {}/{} ({:.3f}%)'.format(correct, len(train_loader.dataset), 100. * correct / (len(train_loader.dataset))))
    logger.info('finish training!')

def test(model, device, val_loader, criterion_mse, criterion_task, logger, target_names):
    model.eval()                                                                                                                                                                                                 
    test_loss = 0
    mse_loss1 = 0
    mse_loss2 = 0
    correct = 0
    logger.info('testing on dev_set')
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)  

    with torch.no_grad():
        for data in tqdm(val_loader):
            spec_n, spec_e, spec, clean_ebd, label = data
            spec_n, spec_e, spec, clean_ebd, label = spec_n.to(device), spec_e.to(device), spec.to(device), clean_ebd.to(device), label.to(device)
            output, noise_ebd, spec_new, _, proj_ebd = model(spec_n,spec_e)
            test_loss += criterion_task(output, label).item()
            mse_loss1 += criterion_mse(spec_new, spec).item()
            mse_loss2 += criterion_mse(proj_ebd, clean_ebd).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            pred = output.data.max(1)[1].cpu().numpy()
            true = label.data.cpu().numpy()
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)

    test_loss /= len(val_loader.dataset)
    mse_loss1 /= len(val_loader)
    mse_loss2 /= len(val_loader)
    acc = 100. * correct / len(val_loader.dataset)

    logger.info('Test set: Average loss_task: {:.4f}, loss_mse1: {:.4f}, loss_mse2: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, mse_loss1, mse_loss2, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)
    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    UA = recall_score(true_all,pred_all,average='macro')     
    WA = recall_score(true_all,pred_all,average='weighted')
    return test_loss,UA,WA

def early_stopping(model,savepath,metricsInEpochs,gap):
    best_metric_inx=np.argmax(metricsInEpochs)
    if best_metric_inx+1==len(metricsInEpochs):
        model_best = os.path.join(savepath, 'model_best_epoch_{}.pt'.format(best_metric_inx+1))
        torch.save(model,model_best)
        return False
    elif (len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else:
        return False

def main():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    seed = args.seed
    max_len = args.max_len
    dataset_type = args.dataset_type
    root = args.root
    fold = args.fold
    loss_type = args.loss_type
    optimizer_type = args.optimizer_type
    device_number = args.device_number
    os.environ["CUDA_VISIBLE_DEVICES"] = device_number

    setup_seed(seed)
    #lr_min = lr * 1e-4
    stamp = datetime.datetime.now().strftime('%y%m%d%H%M')
    tag = stamp + '_' + str(epochs)
    #savedir = os.path.join(root, tag) 
    savedir = os.path.join(root, 'fold_{}'.format(fold))
    try:
        os.makedirs(savedir)
    except OSError:
        if not os.path.isdir(savedir):
            raise

    subprocess.check_call(["cp", "model.py", savedir])
    subprocess.check_call(["cp", "train.py", savedir])
    subprocess.check_call(["cp", "data.py", savedir])
    subprocess.check_call(["cp", "utils.py", savedir])
    subprocess.check_call(["cp", "run.sh", savedir])
    subprocess.check_call(["cp", "augment_cpu.py", savedir])
    logpath = savedir + "/exp.log"
    modelpath = savedir + "/model.pt"
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se) 
    dev_set = Mydataset(dataset_type=dataset_type, mode='val', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se)
    val_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se)

    drop_last = True if len(train_set)%batch_size<2 else False
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    
    model = Mymodel(feature_size=80, h_dims=256, emotion_cls=train_set.NumClasses, backbone_type=args.backbone_type).to(device)

    if loss_type == 'CE':
        #criterion = nn.NLLLoss(weight=train_set.weight.to(device))
        criterion = nn.NLLLoss()
        criterion_test = nn.NLLLoss(reduction='sum')
    elif loss_type == 'Focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.)
        criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    else:
        raise NameError

    criterion_mse = MSE()

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        raise NameError

    logger = get_logger(logpath)
    logger.info(args)
    logger.info('train_set speaker names: {}'.format(train_set.SpeakerNames))
    logger.info('val_set speaker names: {}'.format(dev_set.SpeakerNames))
    logger.info('test speaker names: {}'.format(val_set.SpeakerNames))    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.1, verbose=True)

    val_UA_list = []
    test_UA_dic = {}
    test_WA_dic = {}

    for epoch in range(1, epochs+1):
        start = time.time()
        train(model, device, train_loader, criterion_mse, criterion, optimizer, epoch, logger)   
        val_loss,val_UA,_ = test(model, device, dev_loader, criterion_mse, criterion_test, logger, train_set.ClassNames)
        test_loss,test_UA,test_WA = test(model, device, val_loader, criterion_mse, criterion_test, logger, train_set.ClassNames)
        end = time.time()
        duration = end-start
        val_UA_list.append(val_UA)
        if early_stopping(model,savedir,val_UA_list,gap=50):
            break
        test_UA_dic[test_UA] = epoch
        test_WA_dic[test_WA] = epoch
        scheduler.step(val_UA)
        logger.info("-"*50)
        logger.info('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        logger.info("-"*50)
        time.sleep(0.003)

    best_UA=max(test_UA_dic.keys())
    best_WA=max(test_WA_dic.keys())
    logger.info('UA dic: {}'.format(test_UA_dic))
    logger.info('WA dic: {}'.format(test_WA_dic))    
    logger.info('best UA: {}  @epoch: {}'.format(best_UA,test_UA_dic[best_UA]))
    logger.info('best WA: {}  @epoch: {}'.format(best_WA,test_WA_dic[best_WA]))   
    torch.save(model, modelpath)

if __name__ == '__main__':
    main()
