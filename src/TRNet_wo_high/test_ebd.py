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

from data_ebd import Mydataset
from model import FocalLoss, CNN, RNN, MLP, DNN, CRNN
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score,f1_score

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--seed",default=2021,type=int)
parser.add_argument("--device-number",default='0',type=str)
parser.add_argument('--max-len',default=5,type=int)
parser.add_argument("--feature",type=str,required=True)
args = parser.parse_args()
seed = args.seed

def setup_seed(seed=2021):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

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

def pad_collate(batch):
    #(xx1,xx2,y) = zip(*batch)
    xx1,y = zip(*batch)
    x_lens1 = torch.tensor([x.shape[0] for x in xx1])
    x_lens1, perm_idx1 = x_lens1.sort(descending=True)
    xx_pad1 = pad_sequence(xx1, batch_first=True)
    xx_pad1 = xx_pad1[perm_idx1]
    xx_pack1 = pack_padded_sequence(xx_pad1,x_lens1,batch_first=True,enforce_sorted=False)
    '''
    x_lens2 = torch.tensor([x.shape[0] for x in xx2])
    xx_pad2 = pad_sequence(xx2, batch_first=True)
    xx_pad2 = xx_pad2[perm_idx1]
    x_lens2 = x_lens2[perm_idx1]
    xx_pack2 = pack_padded_sequence(xx_pad2,x_lens2,batch_first=True,enforce_sorted=False)
    '''
    y_emo = np.array(list(y))[perm_idx1]
    return xx_pack1, y_emo

def test(model, device, val_loader, criterion, logger, target_names):
    model = model.to(device)
    model.eval()                                                                                                                                                                                                   
    test_loss = 0
    correct = 0
    true_all = np.array([])
    embedding_ls = []                                                                                                                                                                                                                      
    #aug = Augment(False).to(device)
                                                                                                                                                                                                                   
    with torch.no_grad():
        for spec, label in tqdm(val_loader):
            spec = spec.to(device)
            output,embedding = model(spec)                                                                                                                                                                          
            true_all = np.append(true_all,label)
            embedding_ls.append(embedding.cpu().numpy())                                                                                                                                                          
                                                                                                                                                                                                                   
    embedding_all = np.concatenate(embedding_ls,0)        
                                                                                                                                                                                                                   
    return embedding_all,true_all

def test_epoch(model_root, fold, logger, model_name, max_len, feature):   
    feature_size_dic = {'mfcc':39,'FBank':80,'egemaps':88,'WavLM':768}  
    model_dic = {'mfcc':CRNN,'FBank':CNN,'egemaps':MLP,'WavLM':RNN}    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_type = "IEMOCAP_4"
    batch_size = 4
    feature_ls = feature.split(',')   

    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len=max_len, fold=fold, feature_ls=feature_ls)
    test_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, feature_ls=feature_ls)  
    
    if len(feature_ls)==1:
        model = model_dic[feature_ls[0]](feature_size=[feature_size_dic[feature] for feature in feature_ls], emotion_cls=test_set.NumClasses, h_dims=32, dropout=0.25)
    
    ckpt = torch.load(model_root)
    model.load_state_dict(ckpt)                                                                                                                                                            

    target_names = test_set.ClassNames

    if len(feature_ls)==1 and feature_ls[0] in ['WavLM','mfcc']:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)  
                                                                                                                                                                                                                   
    #criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    criterion_test = nn.NLLLoss(reduction='sum')
                                                                                                                                                                                                                   
    #logger = get_logger('log/exp.log')
    logger.info('testing {}'.format(model_name))
    embedding_all_train, true_all_train = test(model, device, train_loader, criterion_test, logger, target_names)
    embedding_all_test, true_all_test = test(model, device, test_loader, criterion_test, logger, target_names)
                                                                                                                                       
    return embedding_all_train, true_all_train, embedding_all_test, true_all_test

def main(fold_list, fold_root, max_len, feature):
    subprocess.check_call(["cp", "test_ebd.py", fold_root])
    #logpath = os.path.join(fold_root, "{}_test.log".format(condition))
    logpath = os.path.join(fold_root, "test_ebd.log")
    logger = get_logger(logpath)
   
    for fold in fold_list:
        logger.info('fold: {}'.format(fold))
        root = os.path.join(fold_root, 'fold_{}'.format(fold))
        #sub_fold = list(filter(lambda n:n.startswith('condition_'+condition), os.listdir(root)))[0]
        #model_name = list(filter(lambda n:n.startswith('bestmodel-'), os.listdir(os.path.join(root,sub_fold))))[0]
        model_ls = list(i for i in os.listdir(root) if i.startswith('best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        embedding_all_train, true_all_train, embedding_all_test, true_all_test = test_epoch(model_root, fold, logger, best_model, max_len, feature)
        
        #test_epoch(model_root, condition, fold, logger, best_model)
        np.save(os.path.join(root,'embedding_train.npy'),embedding_all_train)
        np.save(os.path.join(root,'fn_train.npy'),true_all_train)
        np.save(os.path.join(root,'embedding_test.npy'),embedding_all_test)
        np.save(os.path.join(root,'fn_test.npy'),true_all_test)        

if __name__ == '__main__':
    #condition = 'impro'
    device_number = args.device_number
    feature = args.feature
    max_len = args.max_len
    os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    fold_list = list(range(1))
    max_len_unit = str(max_len)+'s'
    fold_root = './seed_{}+batchnorm/IEMOCAP_4+{}+{}+lr1e-3+batch_size64+CE+Adam'.format(seed, max_len_unit, feature)
    main(fold_list, fold_root, max_len, feature)