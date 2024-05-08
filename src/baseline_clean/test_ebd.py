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
import pickle
from data_ebd import Mydataset
from model import FocalLoss, Mymodel
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
#from augment_cpu import Spectrogram, MelScale, MelSpectrogram, AxisMasking, FrequencyMasking, TimeMasking, AddNoise, Crop, TimeStretch, PitchShift, Deltas_Deltas, TimeShift
from utils import mixup_data, mixup_criterion
import argparse
from sklearn.metrics import confusion_matrix,classification_report,recall_score,f1_score

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--seed",default=2021,type=int)
parser.add_argument("--device_number",default='0',type=str)
parser.add_argument('--max_len',default=5,type=int)
parser.add_argument("--feature",type=str,required=True)
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--lr',default=1e-3,type=str)
parser.add_argument("--optimizer_type",default='SGD',type=str)
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument("--backbone_type",default='CNN6',type=str)
parser.add_argument('--snr',default=100,type=int)
parser.add_argument('--noise_type',default='ESC50',type=str)
parser.add_argument('--do_se', action='store_true')

args = parser.parse_args()

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

def test(model, device, val_loader, criterion, logger, target_names):
    model = model.to(device)
    model.eval()
    embedding_ls = []
    distribution_ls = []
    fn_all = np.array([])

    with torch.no_grad():
        for spec, fn in tqdm(val_loader):
            spec = spec.to(device)
            output,embedding = model(spec)
            distribution = torch.exp(output)
            embedding_ls.append(embedding.cpu().numpy())
            distribution_ls.append(distribution.cpu().numpy())
            fn_all = np.append(fn_all,fn)

    embedding_all = np.concatenate(embedding_ls,0)
    distribution_all = np.concatenate(distribution_ls,0)
    
    embedding_dic = {}
    distribution_dic = {}
    for i in range(len(fn_all)):
        embedding_dic[fn_all[i]] = embedding_all[i]
        distribution_dic[fn_all[i]] = distribution_all[i]

    return {'embedding_dic':embedding_dic, 'distribution_dic':distribution_dic}

def test_epoch(model_root, fold, seed, logger, model_name, max_len, feature):      
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_type = args.dataset_type
    batch_size = args.batch_size 
    train_set = Mydataset(dataset_type=dataset_type, mode='train', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se)
    val_set = Mydataset(dataset_type=dataset_type, mode='val', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se) 
    test_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se)
    model = Mymodel(feature_size=80, h_dims=256, emotion_cls=test_set.NumClasses, backbone_type=args.backbone_type)
    
    #ckpt = torch.load(model_root)
    #model.load_state_dict(ckpt)
    model = torch.load(model_root)

    target_names = test_set.ClassNames
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    criterion_test = nn.NLLLoss(reduction='sum')

    logger.info('testing {}'.format(model_name))
    res_train = test(model, device, train_loader, criterion_test, logger, target_names)
    res_val = test(model, device, val_loader, criterion_test, logger, target_names)
    res_test = test(model, device, test_loader, criterion_test, logger, target_names)
    return res_train,res_val,res_test

def main(fold_list, fold_root, max_len, feature, seed):
    subprocess.check_call(["cp", "test.py", fold_root])
    logpath = os.path.join(fold_root, "test.log")
    logger = get_logger(logpath)

    AR_ls_ls_test = []
    UAR_ls_test = []
    WAR_ls_test = []
    F1_ls_ls_test = []
    UF1_ls_test = []
    WF1_ls_test = []

    pred_global_test = np.array([],dtype=np.long)
    true_global_test = np.array([],dtype=np.long)

    for fold in fold_list:
        logger.info('fold: {}'.format(fold))
        root = os.path.join(fold_root, 'fold_{}'.format(fold))
        model_ls = list(i for i in os.listdir(root) if i.startswith('best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        res_train,res_val,res_test = test_epoch(model_root, fold, seed, logger, best_model, max_len, feature)
        with open(os.path.join(root,'embedding_train.pkl'),'wb') as f:
            pickle.dump(res_train['embedding_dic'],f)
        with open(os.path.join(root,'distribution_train.pkl'),'wb') as f:
            pickle.dump(res_train['distribution_dic'],f)            

        with open(os.path.join(root,'embedding_val.pkl'),'wb') as f:
            pickle.dump(res_val['embedding_dic'],f)
        with open(os.path.join(root,'distribution_val.pkl'),'wb') as f:
            pickle.dump(res_val['distribution_dic'],f)  

        with open(os.path.join(root,'embedding_test.pkl'),'wb') as f:
            pickle.dump(res_test['embedding_dic'],f)
        with open(os.path.join(root,'distribution_test.pkl'),'wb') as f:
            pickle.dump(res_test['distribution_dic'],f)              

if __name__ == '__main__':
    device_number = args.device_number
    feature = args.feature
    max_len = args.max_len
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_number
    if args.dataset_type.startswith('IEMOCAP'):
        fold_list = list(range(10))
    elif args.dataset_type.startswith('MSP-IMPROV'):
        fold_list = list(range(12))
    max_len_unit = str(max_len)+'s'
    fold_root = './seed_{}/{}+{}+{}+lr{}+batch_size{}+CE+{}'.format(seed,args.dataset_type,max_len_unit,feature,args.lr,args.batch_size,args.optimizer_type)
    main(fold_list, fold_root, max_len, feature, seed)