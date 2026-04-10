import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union

class CRN(nn.Module):
    """
    Convolutional Recurrent Network for speech enhancement.
    Based on: "A Convolutional Recurrent Neural Network for Real-Time Speech Enhancement"
    (Tan & Wang, 2018) with optional optimizations.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 fft_point: int = 320,
                 enc_channels: List[int] = [16, 32, 64, 128, 256],
                 enc_kernels: List[int] = [2, 3],
                 enc_strides: List[int] = [1, 2],
                 rnn_hidden_size: int = 256,
                 rnn_layers: int = 2,
                 rnn_bidirectional: bool = False,
                 dec_channels: List[int] = None,   # if None, automatically symmetric
                 dec_kernels: List[int] = None,
                 dec_strides: List[int] = None,
                 activation: str = 'elu'):         # 'relu' or 'elu'
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        if dec_channels is None:
            dec_channels = list(reversed(enc_channels))
        if dec_kernels is None:
            dec_kernels = enc_kernels
        if dec_strides is None:
            dec_strides = enc_strides

        self.encoders = nn.ModuleList()
        for i, out_ch in enumerate(enc_channels):
            in_ch = in_channels if i == 0 else enc_channels[i-1]
            self.encoders.append(self._conv_block(in_ch, out_ch, enc_kernels, enc_strides))

        self.rnn_input_size = enc_channels[-1] * ((fft_point // 64) + 1)
        self.rnn = nn.LSTM(input_size=self.rnn_input_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True,
                           bidirectional=rnn_bidirectional)

        self.rnn_recover = nn.Linear(rnn_hidden_size, self.rnn_input_size)

        self.decoders = nn.ModuleList()
        for i in range(len(dec_channels)):
            out_ch = in_channels if i == (len(dec_channels)-1) else dec_channels[i+1]
            is_last = 1 if i==(len(dec_channels)-1) else 0
            self.decoders.append(self._deconv_block(dec_channels[i]*2, out_ch, dec_kernels, dec_strides, is_last))

    def _conv_block(self, in_ch, out_ch, kernel, stride):
        """Encoder block: Conv2d -> BN -> Activation"""
        padding = kernel // 2 if isinstance(kernel, int) else [k//2 for k in kernel]
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_ch)
        ]
        if self.activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.activation == 'elu':
            layers.append(nn.ELU(inplace=True))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return nn.Sequential(*layers)

    def _deconv_block(self, in_ch, out_ch, kernel, stride, is_last):
        """Decoder block: ConvTranspose2d -> BN -> Activation"""
        padding = kernel // 2 if isinstance(kernel, int) else [k//2 for k in kernel]
        # output_padding needed when stride>1 and input size may not be exactly divisible
        output_padding = (stride - 1) if isinstance(stride, int) else [s-1 for s in stride]
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                               padding=padding, output_padding=(0, 0)),
            nn.BatchNorm2d(out_ch)
        ]
        if is_last:
            layers.append(nn.ReLU(inplace=True))
        else:
            if self.activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif self.activation == 'elu':
                layers.append(nn.ELU(inplace=True))
        
        return nn.Sequential(*layers)

    def _crop(self, tensor, target_shape):
        """Center crop tensor to match target_shape (H, W)"""
        _, _, h, w = tensor.shape
        target_h, target_w = target_shape
        if h == target_h and w == target_w:
            return tensor
        dh = (h - target_h) // 2
        dw = (w - target_w) // 2
        return tensor[:, :, dh:dh+target_h, dw:dw+target_w]

    def forward(self, x):
        """
        Input: (B, C, F, T)  magnitude spectrogram
        Output: (B, C, F, T) enhanced magnitude spectrogram
        """
        _, _, F_in, T_in = x.shape
        x = x.permute(0, 1, 3, 2).contiguous()

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)

        rnn_out, _ = self.rnn(x)
        rnn_out = self.rnn_recover(rnn_out)

        rnn_out = rnn_out.view(B, T, -1, 1)
        C_rnn = rnn_out.shape[2]

        rnn_out = rnn_out.view(B, T, C, F)
        x = rnn_out.permute(0, 2, 1, 3).contiguous()

        for i, dec in enumerate(self.decoders):
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = dec(x)
        
        x = x.permute(0, 1, 3, 2).contiguous()
        out = self._crop(x, (F_in, T_in))
        return out