import warnings
warnings.filterwarnings('ignore')
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
from models import generator
from utils import *
#import torch.multiprocessing as mp
#torch.multiprocessing.set_start_method('spawn')
import argparse

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--dataset_type",type=str,required=True)
parser.add_argument("--device_number",default='0',type=str)
parser.add_argument('--max_len',default=5,type=int)
parser.add_argument('--snr',default=100,type=int)
parser.add_argument('--seed',default=42,type=int)
parser.add_argument('--noise_type',default='ESC50',type=str)
parser.add_argument('--do_se', action='store_true')
args = parser.parse_args()

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

def get_audio_dir_path_IEM(filename):
    # Ses02F_script03_1_F006
    '''
    session = 'Session'+filename.split('_')[0][-2]
    dialog = '_'.join(filename.split('_')[:-1])
    audio_dir = os.path.join('/home/chenchengxin/dataset/IEMOCAP_full_release', session,'sentences/wav',dialog,'{}.wav'.format(filename))
    '''
    audio_dir = os.path.join('/home/chenchengxin/dataset/IEMOCAP_full_release_audio','{}.wav'.format(filename))
    return audio_dir

def get_audio_dir_path_MSP(filename):
    # MSP-IMPROV-S01H-F02-R-FF01
    '''
    session = 'session'+filename.split('-')[-3][-1]
    dialog = filename.split('-')[-4]
    stage = filename.split('-')[-2]
    audio_dir = os.path.join('/home/chenchengxin/dataset/MSP-IMPROV', 'Audio', session, dialog, stage,'{}.wav'.format(filename))
    '''
    audio_dir = os.path.join('/home/chenchengxin/dataset/MSP-IMPROV_audio','{}.wav'.format(filename))
    return audio_dir    

def enhance_one_track(model, wav, cut_len, n_fft=400, hop=100, device='cpu'):
    noisy = wav.to(device)
    c = torch.sqrt(noisy.size(-1) / torch.sum((noisy**2.0), dim=-1))
    noisy = torch.transpose(noisy, 0, 1)
    noisy = torch.transpose(noisy * c, 0, 1)

    length = noisy.size(-1)
    frame_num = int(np.ceil(length / 100))
    padded_len = frame_num * 100
    padding_len = padded_len - length
    noisy = torch.cat([noisy, noisy[:, :padding_len]], dim=-1)
    if padded_len > cut_len:
        batch_size = int(np.ceil(padded_len / cut_len))
        while 100 % batch_size != 0:
            batch_size += 1
        noisy = torch.reshape(noisy, (batch_size, -1))

    noisy_spec = torch.stft(
        noisy, n_fft, hop, window=torch.hamming_window(n_fft).to(device), onesided=True
    )
    noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
    with torch.no_grad():
        est_real, est_imag = model(noisy_spec)
    est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)

    est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
    est_audio = torch.istft(
        est_spec_uncompress,
        n_fft,
        hop,
        window=torch.hamming_window(n_fft).to(device),
        onesided=True,
    )
    est_audio = est_audio / c
    est_audio = torch.flatten(est_audio)[:length].cpu()
    assert len(est_audio) == length
    return est_audio.unsqueeze(0)
    
class Mydataset(Dataset):
    def __init__(self, dataset_type='IEMOCAP_4', max_len=6, fold=1, snr=100, seed=42, noise_type='ESC50', device=torch.device('cuda'), do_se=False):
        self.snr = snr
        self.do_se = do_se
        if self.do_se:
            self.device = device
            self.n_fft = 400
            self.cut_len = 16000*max_len
            self.model = generator.TSCNet(num_channel=64, num_features=self.n_fft // 2 + 1).to(device)
            self.model.load_state_dict((torch.load('./best_ckpt/ckpt')))
            self.model.eval()

        df_tmp = pd.read_csv('/home/chenchengxin/noise_SER/meta/{}.tsv'.format(dataset_type), sep='\t')
        self.data_info = df_tmp[df_tmp.fold==fold].reset_index(drop=True)

        if dataset_type=='MSP-IMPROV':
            self.get_audio_dir_path = get_audio_dir_path_MSP
        else:
            self.get_audio_dir_path = get_audio_dir_path_IEM
        self.pad_trunc = Pad_trunc_wav(max_len*16000)    
        self.transform = Deltas_Deltas_FBank()

        if noise_type=='ESC50':
            df_tmp = pd.read_csv('/home/chenchengxin/noise_SER/meta/{}.tsv'.format(noise_type),sep='\t')
            #df_tmp = df_tmp[df_tmp.fold==fold].reset_index(drop=True)
            self.noise_df = df_tmp.sample(n=len(self.data_info),replace=True,random_state=seed).reset_index(drop=True)
            self.noise_path = '/home/chenchengxin/dataset/ESC-50_audio/'
        elif noise_type=='MUSAN':
            df_tmp = pd.read_csv('/home/chenchengxin/noise_SER/meta/{}.tsv'.format(noise_type),sep='\t')
            #df_tmp = df_tmp[df_tmp.fold==fold].reset_index(drop=True)
            self.noise_df = df_tmp.sample(n=len(self.data_info),replace=False,random_state=seed+fold).reset_index(drop=True)
            self.noise_path = '/home/chenchengxin/dataset/MUSAN_audio/'           

        self.label = self.data_info['label'].astype('category').cat.codes.values                                                                                                                                                                             
        self.ClassNames = np.unique(self.data_info['label'])
        self.NumClasses = len(self.ClassNames)
        self.weight = 1/torch.tensor([(self.label==i).sum() for i in range(self.NumClasses)]).float()

    def pre_process(self, wav):
        wav = self.pad_trunc(wav)
        return wav

    def extract_fbank(self, wav, sample_rate):
        wav = wav * (1 << 15)
        if sample_rate!=16000:
            wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wav)
        spec = kaldi.fbank(wav,num_mel_bins=80,frame_length=25,frame_shift=10,sample_frequency=16000,high_freq=8000,low_freq=0,window_type='hamming')
        return spec

    def add_noise(self, speech, noise):
        power_speech = (speech ** 2).mean()
        power_noise = (noise ** 2).mean()
        scale = (10 ** (-self.snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10)))
        speech = speech + scale * noise    
        return speech

    def __len__(self):
        return len(self.data_info)
                                                                                                                                                                                                                   
    def __getitem__(self, idx):
        wav, sample_rate = torchaudio.load(self.get_audio_dir_path(self.data_info.filename[idx]))
        wav = self.pre_process(wav)
        if self.snr<100:
            noise, sample_rate_noise = torchaudio.load(self.noise_path+self.noise_df.filename[idx])
            assert  sample_rate_noise == sample_rate
            noise = self.pre_process(noise)
            wav = self.add_noise(wav, noise)
        if self.do_se:     
            wav = enhance_one_track(self.model, wav, self.cut_len, self.n_fft, self.n_fft // 4, self.device)
            #wav = mp.Process(target=enhance_one_track,args=(self.model, wav, self.cut_len, self.n_fft, self.n_fft // 4, self.device))

        return wav, self.data_info.filename[idx]

def extract(dset,path,fold):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Processing fold={} at {}'.format(fold,path))        
    for i in range(len(dset)):
        wav, fn = dset[i]
        wav_path = os.path.join(path,fn+'.wav')
        torchaudio.save(wav_path,wav,sample_rate=16000)
    return True

def main(max_len, snr, noise_type, do_se, dataset_type, seed, fold):
    root = '/home/chenchengxin/noise_SER'
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")    
    dset = Mydataset(dataset_type=dataset_type, max_len=max_len, fold=fold, seed=seed, snr=snr, noise_type=noise_type, device=device, do_se=do_se)
    path = os.path.join(root,'{}-SE={}-LOSO'.format(dataset_type,do_se),'seed_{}'.format(seed),'maxlen{}-{}-SNR{}'.format(max_len,noise_type,snr))
    extract(dset,path,fold)

if __name__ == '__main__':
    device_number = args.device_number
    max_len = args.max_len
    snr = args.snr
    noise_type = args.noise_type
    do_se = args.do_se
    dataset_type = args.dataset_type
    seed = args.seed
    os.environ["CUDA_VISIBLE_DEVICES"] = device_number
    fold_num = 5 if dataset_type=='IEMOCAP_4' else 6
    for fold in range(1,fold_num+1):
        main(max_len, snr, noise_type, do_se, dataset_type, seed, fold)
