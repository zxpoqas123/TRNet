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
from model import FocalLoss, Mymodel
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
parser.add_argument('--batch-size',default=16,type=int)
parser.add_argument('--lr',default=1e-3,type=str)
parser.add_argument("--optimizer-type",default='SGD',type=str)
parser.add_argument("--dataset-type",type=str,required=True)
parser.add_argument("--condition",default='all',type=str)
parser.add_argument("--backbone-type",default='CNN6',type=str)

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
    y_emo = torch.tensor(list(y))[perm_idx1]
    return xx_pack1, y_emo

def test(model, device, val_loader, criterion, logger, target_names):
    model = model.to(device)
    model.eval()                                                                                                                                                                                                   
    test_loss = 0
    correct = 0
    pred_all = np.array([],dtype=np.long)
    true_all = np.array([],dtype=np.long)
    embedding_ls = []                                                                                                                                                                                                                      
    #aug = Augment(False).to(device)
                                                                                                                                                                                                                   
    with torch.no_grad():
        for spec, label in tqdm(val_loader):
            spec, label = spec.to(device), label.to(device)
            #mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=n_fft, f_min=0.0, f_max=sr/2, n_mels=n_mels)(wav)
            #mel_specgram = torch.log2(mel_specgram+1e-8)
            #mel_specgram, label = mel_specgram.to(device), label.to(device)
            #output = model(mel_specgram)
            #spec = aug(wav)
            output,embedding = model(spec)                                                                                                                                                                          
            #test_loss += F.nll_loss(output, label, reduction='sum').item()
            test_loss += criterion(output, label).item()                                                                                                                                                           
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            pred = output.data.max(1)[1].cpu().numpy()  
            true = label.data.cpu().numpy()            
            pred_all = np.append(pred_all,pred)
            true_all = np.append(true_all,true)
            embedding_ls.append(embedding.cpu().numpy())                                                                                                                                                          
                                                                                                                                                                                                                   
    test_loss /= len(val_loader.dataset)
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
    
    speaker = val_loader.dataset.speaker                                                                                                                                                                               
    speaker = np.array(speaker).astype('long')   
    
    res = {'AR_ls':AR_ls,
        'UAR':UAR,
        'WAR':WAR,
        'F1_ls':F1_ls,
        'UF1':UF1,
        'WF1':WF1,
        'embedding_all':embedding_all,
        'true_all':true_all,
        'pred_all':pred_all,
        'speaker':speaker}   
    return res

def test_epoch(model_root, fold, logger, model_name, max_len, feature):   
    feature_size_dic = {'spectrogram':257,'mfcc':39,'FBank':80,'egemaps':88,'compare':6373,'WavLM':768}     
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    dataset_type = args.dataset_type
    batch_size = args.batch_size
    feature_ls = feature.split(',')   
    condition = args.condition
    #dev_set = Mydataset(dataset_type=dataset_type, mode='val', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)
    test_set = Mydataset(dataset_type='MSP-IMPROV', mode='all', max_len=max_len, fold=fold, feature_ls=feature_ls, condition=condition)  
    
    if len(feature_ls)==1:
        model = Mymodel(feature_size=[feature_size_dic[feature] for feature in feature_ls], h_dims=128, emotion_cls=test_set.NumClasses, backbone_type=args.backbone_type)
        
    model = torch.load(model_root)                                                                                                                                                           

    target_names = test_set.ClassNames

    if len(feature_ls)==1 and feature_ls[0] in ['WavLM','mfcc']:
        #dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,collate_fn=pad_collate)
    else:
        #dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)  
                                                                                                                                                                                                                   
    #criterion_test = FocalLoss(alpha=0.25, gamma=2., reduction='sum')
    criterion_test = nn.NLLLoss(reduction='sum')
                                                                                                                                                                                                                   
    #logger = get_logger('log/exp.log')
    logger.info('testing {}'.format(model_name))
    #res_dev = test(model, device, dev_loader, criterion_test, logger, target_names)
    res_test = test(model, device, test_loader, criterion_test, logger, target_names)
                                                                                                                                       
    return target_names,res_test

def main(fold_list, fold_root, max_len, feature):
    subprocess.check_call(["cp", "test.py", fold_root])
    #logpath = os.path.join(fold_root, "{}_test.log".format(condition))
    logpath = os.path.join(fold_root, "test_cross.log")
    logger = get_logger(logpath)

    AR_ls_ls_test = []
    UAR_ls_test = []
    WAR_ls_test = []
    F1_ls_ls_test = []
    UF1_ls_test = []
    WF1_ls_test = []

    for fold in fold_list:
        logger.info('fold: {}'.format(fold))
        root = os.path.join(fold_root, 'fold_{}'.format(fold))
        #sub_fold = list(filter(lambda n:n.startswith('condition_'+condition), os.listdir(root)))[0]
        #model_name = list(filter(lambda n:n.startswith('bestmodel-'), os.listdir(os.path.join(root,sub_fold))))[0]
        model_ls = list(i for i in os.listdir(root) if i.startswith('best_epoch'))
        epoch_ls = [int(i[:-3].split('_')[-1]) for i in model_ls]
        best_model = model_ls[np.argmax(epoch_ls)]
        model_root = os.path.join(root,best_model)
        target_names,res_test = test_epoch(model_root, fold, logger, best_model, max_len, feature)
        
        #test_epoch(model_root, condition, fold, logger, best_model)
        np.save(os.path.join(root,'embedding_test_MSP.npy'),res_test['embedding_all'])
        np.save(os.path.join(root,'label_test_MSP.npy'),res_test['true_all'])
        np.save(os.path.join(root,'pred_test_MSP.npy'),res_test['pred_all'])
        np.save(os.path.join(root,'speaker_test_MSP.npy'),res_test['speaker'])

        AR_ls_ls_test.append(res_test['AR_ls'])
        UAR_ls_test.append(res_test['UAR'])
        WAR_ls_test.append(res_test['WAR'])        
        F1_ls_ls_test.append(res_test['F1_ls'])
        UF1_ls_test.append(res_test['UF1'])
        WF1_ls_test.append(res_test['WF1'])

    #=====================================================================================

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

    with open('./test_all_seed_{}.txt'.format(seed),'a') as f:
        f.write('IEMOCAP2MSP-IMPROV feature:{} max_len:{}\n'.format(feature,max_len))
        f.write('Class Names: {}\n'.format(target_names))

        f.write('AR_ls_test avg: {}\n'.format(np.mean(AR_ls_ls_test,0)))
        f.write('UAR_test avg: {}\n'.format(np.mean(UAR_ls_test)))
        f.write('WAR_test avg: {}\n'.format(np.mean(WAR_ls_test)))        
        f.write('F1_ls_test avg: {}\n'.format(np.mean(F1_ls_ls_test,0)))
        f.write('UF1_test avg: {}\n'.format(np.mean(UF1_ls_test)))
        f.write('WF1_test avg: {}\n'.format(np.mean(WF1_ls_test)))
        f.write('='*40+'\n')


if __name__ == '__main__':
    #condition = 'impro'
    device_number = args.device_number
    feature = args.feature
    max_len = args.max_len
    os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    fold_list = list(range(10))
    max_len_unit = str(max_len)+'s'
    fold_root = './seed_{}/{}+{}+{}+lr{}+batch_size{}+CE+{}'.format(seed,args.dataset_type,max_len_unit,feature,args.lr,args.batch_size,args.optimizer_type)
    main(fold_list, fold_root, max_len, feature)