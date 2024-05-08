import math
from typing import Callable, Optional
import numpy as np
import random
import librosa

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import functional as audioF
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Function
from collections import OrderedDict
#from modules.transformer import TransformerEncoder
from modules.transformer import TransformerEncoder,TransformerEncoderLayer

class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.25, gamma = 2, weight = None, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.nllloss = nn.NLLLoss(weight = weight, reduction = 'none')

    def forward(self, output, label):
        logp = self.nllloss(output, label)
        p = torch.exp(-logp)
        loss = self.alpha * (1-p)**self.gamma * logp
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_
    def set_lambda(self, lambda_new):
        self.lambda_ = lambda_new
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock5x5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class AttBlock(nn.Module):
    def __init__(self, n_in, n_out, activation='linear', temperature=1.):
        super(AttBlock, self).__init__()
        
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        self.cla = nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.bn_att = nn.BatchNorm1d(n_out)
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)
         
    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)        

class CNN6(nn.Module):
    def __init__(self, feature_dims):
        
        super(CNN6, self).__init__()

        # Spec augmenter
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #    freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(feature_dims)

        self.conv_block1 = ConvBlock5x5(in_channels=3, out_channels=64)
        self.conv_block2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock5x5(in_channels=256, out_channels=512)

        #self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        #init_layer(self.fc1)
 
    def forward(self, inputs):
        #Inputs: FBank (batch_size, 3, time_steps, freq_bins)
        
        x = inputs.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        #if self.training:
        #    x = self.spec_augmenter(x)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)  # x: (batch_size, 2048, time_steps//32)
        '''
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)  # x: (batch_size, 2048, time_steps//32)
        x = F.dropout(x, p=0.5, training=self.training)
        '''
        return x

class EmoCLS(nn.Module):
    def __init__(self, emotion_cls=4, h_dims=256):
        super(EmoCLS, self).__init__()
        self.fc1 = nn.Linear(h_dims, h_dims//2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn = nn.BatchNorm1d(h_dims//2)
        self.fc2 = nn.Linear(h_dims//2, emotion_cls)
        self.outlayer = nn.LogSoftmax(dim=-1)

    def init_weight(self):
        init_bn(self.bn)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        x = self.dropout(self.activation(self.bn(self.fc1(x))))
        out = self.outlayer(self.fc2(x))
        return out, x

class Mymodel(nn.Module):
    def __init__(self, feature_size=80, h_dims=256, emotion_cls=4, backbone_type='CNN6'):
        super(Mymodel, self).__init__()
        #self.bn1 = nn.BatchNorm2d(3)
        if backbone_type=='CNN6':
            self.backbone = CNN6(feature_size)
        self.att_block = AttBlock(512, h_dims, activation='linear')
        self.EmotionClassifier = EmoCLS(emotion_cls, h_dims)

    def forward(self, inputs):
        x = self.backbone(inputs)
        embedding, att, _ = self.att_block(x)        
        pred, _ = self.EmotionClassifier(embedding)
        return pred, embedding
