import os
import numpy as np
import pandas as pd
from pandas import Series
import soundfile as sound
import random
import librosa
import pickle
import torch
import torchaudio
torchaudio.set_audio_backend("soundfile")
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from augment_cpu import Pad_trunc_wav, Deltas_Deltas_FBank

class Augment(object):
    def __init__(self, max_len=5, feature='egemaps'):
        super(Augment, self).__init__()
        self.feature = feature
        self.feature_coeff_dic={'spectrogram':100,'mfcc':100,'FBank':100,'egemaps':1,'compare':1,'WavLM':50}
        self.l = int(self.feature_coeff_dic[feature]*max_len)
        self.trunc_seq = Trunc_seq(max_len = self.l)
        self.pad_trunc_seq = Pad_trunc_seq(max_len = self.l)
        self.deltas_deltas_mfcc = Deltas_Deltas_mfcc()
        self.deltas_deltas_fbank = Deltas_Deltas_FBank()

    def __call__(self, x):
        if self.feature in ['egemaps','compare']:
            return x
        elif self.feature=='spectrogram':
            out = self.pad_trunc_seq(x)
            return out
        elif self.feature=='mfcc':
            out = self.trunc_seq(x)
            out = self.deltas_deltas_mfcc(out)
            return out
        elif self.feature=='FBank':
            out = self.pad_trunc_seq(x)
            out = self.deltas_deltas_fbank(out)
            return out
        elif self.feature=='WavLM':
            out = self.trunc_seq(x)
            return out
    
class Mydataset(Dataset):
    def __init__(self, dataset_type='IEMOCAP_4', mode='train', max_len=6, fold=0, seed=42, snr=100, noise_type='ESC50', do_se=False):
        self.root = '/home/chenchengxin/noise_SER'
        self.mode = mode
        self.snr = snr
        self.do_se = do_se
        self.dataset_type = dataset_type
        self.seed = seed
        self.max_len = max_len
        self.fold = fold
        self.noise_type = noise_type
        data_all = pd.read_csv('/home/chenchengxin/noise_SER/meta/{}.tsv'.format(dataset_type), sep='\t')
        SpkNames = np.unique(data_all['speaker'])
        self.data_info = self.split_dataset(data_all, fold, SpkNames)
        self.transform = Deltas_Deltas_FBank()
        self.label = self.data_info['label'].astype('category').cat.codes.values
        self.ClassNames = np.unique(self.data_info['label'])
        self.SpeakerNames = np.sort(np.unique(self.data_info['speaker']))
        self.NumClasses = len(self.ClassNames)
        self.weight = 1/torch.tensor([(self.label==i).sum() for i in range(self.NumClasses)]).float()

    def split_dataset(self, df_all, fold, speakers):
        spk_len = len(speakers)
        #test_idx = np.array(df_all['speaker']==speakers[fold*2%spk_len])+np.array(df_all['speaker']==speakers[(fold*2+1)%spk_len])
        #val_idx = np.array(df_all['speaker']==speakers[(fold*2-2)%spk_len])+np.array(df_all['speaker']==speakers[(fold*2-1)%spk_len])
        #train_idx = True^(test_idx+val_idx)
        #train_idx = True^test_idx
        test_idx = np.array(df_all['speaker']==speakers[fold%spk_len])
        if fold%2==0:
            val_idx = np.array(df_all['speaker']==speakers[(fold+1)%spk_len])
        else:
            val_idx = np.array(df_all['speaker']==speakers[(fold-1)%spk_len])
        train_idx = True^(test_idx+val_idx)
        train_data_info = df_all[train_idx].reset_index(drop=True)
        val_data_info = df_all[val_idx].reset_index(drop=True)
        test_data_info = df_all[test_idx].reset_index(drop=True)
        #val_data_info = test_data_info = df_all[test_idx].reset_index(drop=True)
        if self.mode == 'train':
            data_info = train_data_info
        elif self.mode == 'val':
            data_info = val_data_info
        elif self.mode == 'test':
            data_info = test_data_info
        else:
            data_info = df_all
        return data_info

    def extract_fbank(self, wav, sample_rate):
        wav = wav * (1 << 15)
        if sample_rate!=16000:
            wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
        spec = kaldi.fbank(wav,num_mel_bins=80,frame_length=25,frame_shift=10,sample_frequency=16000,high_freq=8000,low_freq=0,window_type='hamming')
        return spec 

    def get_audio_dir_path(self,fn,snr):
        path = os.path.join(self.root,'{}-SE={}-LOSO'.format(self.dataset_type,self.do_se),'seed_{}'.format(self.seed),'maxlen{}-{}-SNR{}'.format(self.max_len,self.noise_type,snr))
        return os.path.join(path, '{}.wav'.format(fn))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode=='test':
            wav, sample_rate = torchaudio.load(self.get_audio_dir_path(self.data_info.filename[idx],self.snr))
        else:
            if self.snr<100:
                snr = random.sample([100,20,15,10,5,0],1)[0]
                wav, sample_rate = torchaudio.load(self.get_audio_dir_path(self.data_info.filename[idx],snr))
            else:
                wav, sample_rate = torchaudio.load(self.get_audio_dir_path(self.data_info.filename[idx],self.snr))
        spec = self.transform(self.extract_fbank(wav, sample_rate)).float()
        label = self.label[idx]
        label = np.array(label)
        label = label.astype('float').reshape(1)
        label = torch.Tensor(label).long().squeeze()
        return spec, label
