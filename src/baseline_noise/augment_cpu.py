import math
from typing import Callable, Optional
import numpy as np
import random
import librosa
import time

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import functional as audioF
from torchaudio.compliance import kaldi

#def setup_seed(seed=20):
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#    np.random.seed(seed)
#    random.seed(seed)
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.deterministic = True
#
#setup_seed(20)

class AxisMasking(nn.Module):
    __constants__ = ['mask_param', 'axis', 'iid_masks']

    def __init__(self, mask_param: int, axis: int, iid_masks: bool) -> None:
        super(AxisMasking, self).__init__()
        self.mask_param = mask_param
        self.axis = axis
        self.iid_masks = iid_masks

    def forward(self, specgram: Tensor, mask_value: float = 0.) -> Tensor:
        if self.iid_masks and specgram.dim() == 4:
            return audioF.mask_along_axis_iid(specgram, self.mask_param, mask_value, self.axis + 1)
        else:
            return audioF.mask_along_axis(specgram, self.mask_param, mask_value, self.axis)

class FrequencyMasking(AxisMasking):
    def __init__(self, freq_mask_param: int, iid_masks: bool = False) -> None:
        super(FrequencyMasking, self).__init__(freq_mask_param, 1, iid_masks)

class TimeMasking(AxisMasking):
    def __init__(self, time_mask_param: int, iid_masks: bool = False) -> None:
        super(TimeMasking, self).__init__(time_mask_param, 2, iid_masks)

class Crop(nn.Module):
    def __init__(self):
        super(Crop, self).__init__()

    def forward(self, spec, length):
#        spec_out = torch.zeros((spec.shape[0], spec.shape[1], spec.shape[2], length), device=spec.device)
#        for j in range(spec.shape[0]):
#            StartLoc = np.random.randint(0, spec.shape[3]-length) 
#            spec_out[j, :, :, 0:length] = spec[j, :, :, StartLoc:StartLoc+length]

        spec_out = torch.zeros((spec.shape[0], spec.shape[1], length), device=spec.device)
        StartLoc = np.random.randint(0, spec.shape[2]-length)
        spec_out = spec[:, :, StartLoc:StartLoc+length]
        return spec_out

class AddNoise(nn.Module):
    def __init__(self):
        super(AddNoise, self).__init__()
                                                                                                                                                                                                                   
    def forward(self, wav, snr=[5, 10, 20]):
        noise = torch.randn(wav.shape, device=wav.device)                                                                                                                                                          
        Ex = wav.mul(wav).sum()                                                                                                                                                                                    
        En = noise.mul(noise).sum()                                                                                                                                                                                
        s = np.random.choice(snr)                                                                                                                                                                                  
        K = (Ex / ((10 ** (s / 10.)) * En)).sqrt()
        noise *= K                                                                                                                                                                                                 
        wav += noise                                                                                                                                                                                               
        return wav
'''
class Deltas_Deltas(nn.Module):
    def __init__(self):
        super(Deltas_Deltas, self).__init__()
                                                                                                                                                                                                                   
    def deltas(self, x):
        x_out = (x[:, :, 2:] - x[:, :, :-2]) / 10.0
        x_out = x_out[:, :, 1:-1] + (x[:, :, 4:] - x[:, :, :-4]) / 5.0
        return x_out
                                                                                                                                                                                                                   
    def forward(self, x):
        x_deltas = self.deltas(x)                                                                                                                                                                                  
        x_deltas_deltas = self.deltas(x_deltas)                                                                                                                                                                    
        x_out = torch.cat((x[:, :, 4:-4], x_deltas[:, :, 2:-2], x_deltas_deltas), 0)
        return x_out
'''
class Deltas_Deltas_FBank(nn.Module):
    #(…, freq, time)
    def __init__(self):
        super(Deltas_Deltas_FBank, self).__init__()
    def forward(self,x):
        # x: time*freq
        x = x.permute(1,0).unsqueeze(0)
        delta = audioF.compute_deltas(x)
        delta2 = audioF.compute_deltas(delta)
        x_out = torch.cat((x,delta,delta2), 0).permute(0,2,1)
        # x_out: 3*freq*time -> 3*time*freq
        return x_out

class Deltas_Deltas_mfcc(nn.Module):
    #(…, freq, time)
    def __init__(self):
        super(Deltas_Deltas_mfcc, self).__init__()
    def forward(self,x):
        # x: time*freq
        x = x.permute(1,0)
        delta = audioF.compute_deltas(x)
        delta2 = audioF.compute_deltas(delta)
        x_out = torch.cat((x,delta,delta2), 0).permute(1,0)
        # x_out: 3*freq*time -> 3*time*freq
        return x_out

class Pad_trunc_wav(nn.Module):
    def __init__(self, max_len: int = 6*16000):
        super(Pad_trunc_wav, self).__init__()
        self.max_len = max_len
    def forward(self,x):
        shape = x.shape
        length = shape[1]
        if length < self.max_len:
            '''            
            pad_shape = (self.max_len - length,) + shape[1:]
            pad = torch.zeros(pad_shape)
            x_new = torch.cat((x, pad), axis=0)
            '''
            multiple = self.max_len//length+1
            x_tmp = torch.cat((x,)*multiple, axis=1)
            x_new = x_tmp[:,0:self.max_len]
        else:
            x_new = x[:,0:self.max_len]
        return x_new
        
class Pad_trunc_seq(nn.Module):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    def __init__(self, max_len: int = 50):
        super(Pad_trunc_seq, self).__init__()
        self.max_len = max_len
    def forward(self,x):
        shape = x.shape
        length = shape[0]
        if length < self.max_len:
            '''            
            pad_shape = (self.max_len - length,) + shape[1:]
            pad = torch.zeros(pad_shape)
            x_new = torch.cat((x, pad), axis=0)
            '''
            multiple = self.max_len//length+1
            x_tmp = torch.cat((x,)*multiple, axis=0)
            x_new = x_tmp[0:self.max_len,:]
        elif length > self.max_len:
            x_new = x[0:self.max_len,:]
        else:
            x_new = x
        return x_new

class Trunc_seq(nn.Module):
    """Pad or truncate a sequence data to a fixed length.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    def __init__(self, max_len: int = 50):
        super(Trunc_seq, self).__init__()
        self.max_len = max_len
    def forward(self,x):
        shape = x.shape
        length = shape[0]
        if length > self.max_len:
            x_new = x[0:self.max_len,:]
        else:
            x_new = x
        return x_new

class Spectrogram(nn.Module):
    def __init__(self,
                 n_fft: int = 2048,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 pad: int = 0,
                 window_fn: Callable[..., Tensor] = torch.hann_window,                                                                                                                                             
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(Spectrogram, self).__init__()
        self.n_fft = n_fft                                                                                                                                                                                         
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        window = window_fn(self.win_length) if wkwargs is None else window_fn(self.win_length, **wkwargs)
        self.register_buffer('window', window)
        self.pad = pad                                                                                                                                                                                             
        self.power = power                                                                                                                                                                                         
        self.normalized = normalized                                                                                                                                                                               
                                                                                                                                                                                                                   
    def forward(self, waveform: Tensor) -> Tensor:
        return audioF.spectrogram(waveform, self.pad, self.window, self.n_fft, self.hop_length, self.win_length, self.power, self.normalized)

class MelScale(nn.Module):
    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 44100,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels                                                                                                                                                                                       
        self.sample_rate = sample_rate                                                                                                                                                                             
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min                                                                                                                                                                                         
                                                                                                                                                                                                                   
        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)
                                                                                                                                                                                                                   
        fb = torch.empty(0) if n_stft is None else audioF.create_fb_matrix(n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        self.register_buffer('fb', fb)
                                                                                                                                                                                                                   
    def forward(self, specgram: Tensor) -> Tensor:
        shape = specgram.size()                                                                                                                                                                                    
        specgram = specgram.reshape(-1, shape[-2], shape[-1])
                                                                                                                                                                                                                   
        if self.fb.numel() == 0:
            tmp_fb = audioF.create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            self.fb.resize_(tmp_fb.size())                                                                                                                                                                         
            self.fb.copy_(tmp_fb)                                                                                                                                                                                  
                                                                                                                                                                                                                   
        mel_specgram = torch.matmul(specgram.transpose(1,2), self.fb).transpose(1,2)
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])
                                                                                                                                                                                                                   
        return mel_specgram

class MelSpectrogram(nn.Module):
    def __init__(self,
                 sample_rate: int = 44100,
                 n_fft: int = 2048,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 pad: int = 0,
                 n_mels: int = 128,
                 window_fn: Callable[..., Tensor] = torch.hann_window,                                                                                                                                             
                 power: Optional[float] = 2.,
                 normalized: bool = False,
                 wkwargs: Optional[dict] = None) -> None:
        super(MelSpectrogram, self).__init__()
        self.sample_rate = sample_rate                                                                                                                                                                             
        self.n_fft = n_fft                                                                                                                                                                                         
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad                                                                                                                                                                                             
        self.power = power                                                                                                                                                                                         
        self.normalized = normalized                                                                                                                                                                               
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max                                                                                                                                                                                         
        self.f_min = f_min                                                                                                                                                                                         
        self.spectrogram = Spectrogram(n_fft=self.n_fft, win_length=self.win_length,hop_length=self.hop_length,                                                                                                    
                                       pad=self.pad, window_fn=window_fn, power=self.power,                                                                                                                        
                                       normalized=self.normalized, wkwargs=wkwargs)                                                                                                                                
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1)
                                                                                                                                                                                                                   
    def forward(self, waveform: Tensor) -> Tensor:
        specgram = self.spectrogram(waveform)                                                                                                                                                                      
        mel_specgram = self.mel_scale(specgram)                                                                                                                                                                    
        return mel_specgram

class TimeStretch(nn.Module):
    def __init__(self,
                 n_freq: int = 1025,
                 n_fft: int = 2048) -> None:
        super(TimeStretch, self).__init__()
        hop_length = n_fft // 2
        self.register_buffer('phase_advance', torch.linspace(0, math.pi * hop_length, n_freq)[..., None])
        self.spectrogram_c = Spectrogram(n_fft=2048, win_length=2048,hop_length=1024,
                                   pad=0, window_fn=torch.hann_window, power=None,
                                   normalized=False, wkwargs=None)

    def forward(self, wav, rate=1, St=True):
#        if rate == 1:
#            return wav
        complex_specgrams = self.spectrogram_c(wav)
        if St == True:
            rate = np.random.uniform(0.9, 1.1)
            s = audioF.phase_vocoder(complex_specgrams, rate, self.phase_advance)
            wav_n = audioF.istft(s, 2048, 1024, 2048, torch.hann_window(2048, device=complex_specgrams.device), True, 'reflect', False, True)
            if wav_n.shape[-1] < wav.shape[-1]:
                wav_n = torch.cat((wav_n, wav_n), -1)
                return wav_n[:, :, 0:wav.shape[-1]]
            else:
                return wav_n[:, :, 0:wav.shape[-1]]
        else:
            if rate == 1:
                s = complex_specgrams
                wav_n = audioF.istft(s, 2048, 1024, 2048, torch.hann_window(2048, device=complex_specgrams.device), True, 'reflect', False, True, wav.shape[-1])
            else:
                rate = rate
                s = audioF.phase_vocoder(complex_specgrams, rate, self.phase_advance)
                wav_n = audioF.istft(s, 2048, 1024, 2048, torch.hann_window(2048, device=complex_specgrams.device), True, 'reflect', False, True)
            return wav_n

class PitchShift(nn.Module):                                                                                                                                                                                       
    def __init__(self, sample_rate=44100, bins_per_octave=12):                                                                                                                                                     
        super(PitchShift, self).__init__()                                                                                                                                                                         
        self.sample_rate = sample_rate                                                                                                                                                                             
        self.bins_per_octave = bins_per_octave                                                                                                                                                                     
        self.time_stretch = TimeStretch(n_freq=1025, n_fft=2048)                                                                                                                                                   

    def forward(self, wav):                                                                                                                                                                                        
        #n_steps = np.random.uniform(-4, 4)
        n_steps = np.random.choice((0,2))                                                                                                                                                                          
#        if n_steps == 0:
#            return wav
#        n_steps = 0
#        n_steps = np.random.uniform(-1, 1)
        rate = 2.0 ** (-float(n_steps) / self.bins_per_octave)                                                                                                                                                     
        wav_n = self.time_stretch(wav, rate, False)                                                                                                                                                                

        #resample
        shape = wav_n.size()                                                                                                                                                                                       
        wav_n = wav_n.view(-1, shape[-1])                                                                                                                                                                          
        wav_n = kaldi.resample_waveform(wav_n, int(self.sample_rate / rate), self.sample_rate)                                                                                                                     
        wav_n = wav_n.view(shape[:-1] + wav_n.shape[-1:])                                                                                                                                                          

        if wav_n.shape[-1] < wav.shape[-1]:                                                                                                                                                                        
            return F.pad(wav_n, (0,wav.shape[-1]-wav_n.shape[-1]), "constant", 0)                                                                                                                                  
        else:                                                                                                                                                                                                      
            return wav_n[:, :, 0:wav.shape[-1]]

class TimeShift(nn.Module):                                                                                                                                                                                        
    def __init__(self):                                                                                                                                                                                            
        super(TimeShift, self).__init__()                                                                                                                                                                          

    def forward(self, wav):                                                                                                                                                                                        
        wav_n = torch.zeros(wav.shape, device=wav.device)                                                                                                                                                          
#        for j in range(wav.shape[0]):
#            shifts = np.random.randint(-44100, 44100)
#            #shifts = 1
#            wav_n[j,:,:] = torch.roll(wav[j,:,:], shifts, -1)
        shifts = np.random.randint(-44100, 44100)                                                                                                                                                                  
        wav_n = torch.roll(wav, shifts, -1)
        return wav_n