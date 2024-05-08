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

from data import Mydataset
from model import FocalLoss, Mymodel, MSE
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
parser.add_argument('--alpha',default=1.0,type=float)
parser.add_argument('--beta',default=1.0,type=float)

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

def test(model, device, val_loader, criterion_mse, criterion_task, logger, target_names):
    model = model.to(device)
    model.eval()
    test_loss = 0
    mse_loss1 = 0
    mse_loss2 = 0
    correct = 0
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)
    embedding_ls = []
    S_score_ls = []

    with torch.no_grad():
        for data in tqdm(val_loader):
            spec_n, spec_e, spec, clean_ebd, label = data
            spec_n, spec_e, spec, clean_ebd, label = spec_n.to(device), spec_e.to(device), spec.to(device), clean_ebd.to(device), label.to(device)
            output, embedding, spec_new, S_score, proj_ebd = model(spec_n,spec_e)
            test_loss += criterion_task(output, label).item()
            mse_loss1 += criterion_mse(spec_new, spec).item()
            mse_loss2 += criterion_mse(proj_ebd, clean_ebd).item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            pred = output.data.max(1)[1].cpu().numpy()
            true = label.data.cpu().numpy()
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)
            embedding_ls.append(proj_ebd.cpu().numpy())
            S_score_ls.append(S_score.cpu().numpy())

    test_loss /= len(val_loader.dataset)
    mse_loss1 /= len(val_loader)
    mse_loss2 /= len(val_loader)
    acc = 100. * correct / len(val_loader.dataset)
                                                                                                                                                                                                                   
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), acc))

    con_mat = confusion_matrix(true_all,pred_all)
    cls_rpt = classification_report(true_all,pred_all,target_names=target_names,digits=3)

    F1_ls = f1_score(true_all,pred_all,average=None)
    UF1 = f1_score(true_all,pred_all,average='macro')
    WF1 = f1_score(true_all,pred_all,average='weighted')

    AR_ls = recall_score(true_all,pred_all,average=None)
    UAR = recall_score(true_all,pred_all,average='macro')
    WAR = recall_score(true_all,pred_all,average='weighted')

    logger.info('Confusion Matrix:\n{}\n'.format(con_mat))
    logger.info('Classification Report:\n{}\n'.format(cls_rpt))
    embedding_all = np.concatenate(embedding_ls,0)
    S_score_all = np.concatenate(S_score_ls,0)
    
    res = {'AR_ls':AR_ls,
        'UAR':UAR,
        'WAR':WAR,
        'F1_ls':F1_ls,
        'UF1':UF1,
        'WF1':WF1,
        'embedding_all':embedding_all,
        'S_score_all':S_score_all,
        'true_all':true_all,
        'pred_all':pred_all}                                                                                                                                                                                       
    return res

def test_epoch(model_root, fold, seed, logger, model_name, max_len, feature):      
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_type = args.dataset_type
    batch_size = args.batch_size  
    test_set = Mydataset(dataset_type=dataset_type, mode='test', max_len=max_len, fold=fold, seed=seed, snr=args.snr, noise_type=args.noise_type, do_se=args.do_se)
    model = Mymodel(feature_size=80, h_dims=256, emotion_cls=test_set.NumClasses, backbone_type=args.backbone_type)

    #ckpt = torch.load(model_root)
    #model.load_state_dict(ckpt)
    model = torch.load(model_root)

    target_names = test_set.ClassNames
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    criterion_test = nn.NLLLoss(reduction='sum')
    criterion_mse = MSE()

    logger.info('testing {}'.format(model_name))
    res_test = test(model, device, test_loader, criterion_mse, criterion_test, logger, target_names)
    return target_names,res_test

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
        model_ls = list(i for i in os.listdir(root) if i.startswith('model_best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        target_names,res_test = test_epoch(model_root, fold, seed, logger, best_model, max_len, feature)

        np.save(os.path.join(root,'embedding_test_{}_snr{}.npy'.format(args.noise_type,args.snr)),res_test['embedding_all'])
        np.save(os.path.join(root,'S_score_test_{}_snr{}.npy'.format(args.noise_type,args.snr)),res_test['S_score_all'])
        np.save(os.path.join(root,'label_test_snr.npy'),res_test['true_all'])
        np.save(os.path.join(root,'pred_test_{}_snr{}.npy'.format(args.noise_type,args.snr)),res_test['pred_all'])

        AR_ls_ls_test.append(res_test['AR_ls'])
        UAR_ls_test.append(res_test['UAR'])
        WAR_ls_test.append(res_test['WAR'])        
        F1_ls_ls_test.append(res_test['F1_ls'])
        UF1_ls_test.append(res_test['UF1'])
        WF1_ls_test.append(res_test['WF1'])

        pred_global_test = np.append(pred_global_test,res_test['pred_all'])
        true_global_test = np.append(true_global_test,res_test['true_all'])

    logger.info('AR_ls_test list: {}'.format(AR_ls_ls_test))
    logger.info('UAR_test list: {}'.format(UAR_ls_test))
    logger.info('WAR_test list: {}'.format(WAR_ls_test))

    logger.info('AR_ls_test avg: {}'.format(np.mean(AR_ls_ls_test,0)))
    logger.info('UAR_test avg: {}'.format(np.mean(UAR_ls_test)))
    logger.info('WAR_test avg: {}'.format(np.mean(WAR_ls_test)))

    logger.info('F1_ls_test list: {}'.format(F1_ls_ls_test))
    logger.info('UF1_test list: {}'.format(UF1_ls_test))
    logger.info('WF1_test list: {}'.format(WF1_ls_test))

    logger.info('F1_ls_test avg: {}'.format(np.mean(F1_ls_ls_test,0)))
    logger.info('UF1_test avg: {}'.format(np.mean(UF1_ls_test)))
    logger.info('WF1_test avg: {}'.format(np.mean(WF1_ls_test)))

    with open('./test_all_seed_{}_{}_alpha{}_beta{}.txt'.format(seed,args.dataset_type,args.alpha,args.beta),'a') as f:
        f.write('feature:{} max_len:{} snr:{} noise_type:{} do_se:{}\n'.format(feature,max_len,args.snr,args.noise_type,args.do_se))
        f.write('fold_list:{}  Class Names: {}\n'.format(fold_list,target_names))

        f.write('AR_ls_test avg: {}\n'.format(np.mean(AR_ls_ls_test,0)))
        f.write('UAR_test avg: {}\n'.format(np.mean(UAR_ls_test)))
        f.write('WAR_test avg: {}\n'.format(np.mean(WAR_ls_test)))        
        f.write('F1_ls_test avg: {}\n'.format(np.mean(F1_ls_ls_test,0)))
        f.write('UF1_test avg: {}\n'.format(np.mean(UF1_ls_test)))
        f.write('WF1_test avg: {}\n'.format(np.mean(WF1_ls_test)))
        f.write('='*40+'\n')

    with open('./test_all_seed_{}_{}_alpha{}_beta{}_global.txt'.format(seed,args.dataset_type,args.alpha,args.beta),'a') as f:
        f.write('feature:{} max_len:{} snr:{} noise_type:{} do_se:{}\n'.format(feature,max_len,args.snr,args.noise_type,args.do_se))
        f.write('fold_list:{}  Class Names: {}\n'.format(fold_list,target_names))

        UAR_test = recall_score(true_global_test,pred_global_test,average='macro')
        WAR_test = recall_score(true_global_test,pred_global_test,average='weighted')
        con_mat = confusion_matrix(true_global_test,pred_global_test)
        cls_rpt = classification_report(true_global_test,pred_global_test,target_names=target_names,digits=4)
        f.write('UAR_test avg: {}\n'.format(UAR_test))
        f.write('WAR_test avg: {}\n'.format(WAR_test))
        f.write('Confusion Matrix:\n{}\n'.format(con_mat))
        f.write('Classification Report:\n{}\n'.format(cls_rpt))
        f.write('='*40+'\n')


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
    fold_root = './seed_{}/alpha{}+beta{}+{}+{}+{}+lr{}+batch_size{}+CE+{}'.format(seed,args.alpha,args.beta,args.dataset_type,max_len_unit,feature,args.lr,args.batch_size,args.optimizer_type)
    main(fold_list, fold_root, max_len, feature, seed)