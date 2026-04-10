import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class ComplexConv2d(nn.Module):
    """
    Input: (B, 2*C, H, W) 
    Output: (B, 2*C, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        
        real_part = self.conv_rr(x_real) - self.conv_ri(x_imag)
        imag_part = self.conv_ir(x_real) + self.conv_ii(x_imag)
        
        return torch.cat([real_part, imag_part], dim=1)

class ComplexTransposeConv2d(nn.Module):
    """
    Input: (B, 2*C, H, W) 
    Output: (B, 2*C, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv_rr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ri = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ir = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ii = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x_real, x_imag = torch.chunk(x, 2, dim=1)
        
        real_part = self.conv_rr(x_real) - self.conv_ri(x_imag)
        imag_part = self.conv_ir(x_real) + self.conv_ii(x_imag)
        
        return torch.cat([real_part, imag_part], dim=1)

class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        
        self.lstm_real = nn.LSTM(input_size, hidden_size, num_layers, 
                                bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.lstm_imag = nn.LSTM(input_size, hidden_size, num_layers,
                                bidirectional=bidirectional, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        real_part, imag_part = torch.chunk(x, 2, dim=-1)
        
        real_out, _ = self.lstm_real(real_part)
        imag_out, _ = self.lstm_imag(imag_part)
        
        return torch.cat([real_out, imag_out], dim=-1)

class DCCRN(nn.Module):
    """
    Deep Complex Convolution Recurrent Network
    Based on: "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"
    (Hu Y & Liu Y & Lv S, 2020)
    """
    def __init__(self, 
                 in_channels=1,
                 out_channels=1,
                 fft_point=512,
                 enc_channels=[32, 64, 128, 256, 256],
                 rnn_hidden_size=256,
                 rnn_layers=2,
                 rnn_bidirectional=True,
                 activation='relu'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        
        self.freq_dim = fft_point // 2 + 1
        
        self.encoders = nn.ModuleList()
        for i, out_ch in enumerate(enc_channels):
            in_ch = in_channels * 2 if i == 0 else enc_channels[i-1] * 2
            self.encoders.append(
                self._complex_conv_block(in_ch, out_ch * 2, kernel_size=(2, 5), stride=(1, 2))
            )
        
        last_enc_dim = enc_channels[-1]
        rnn_input_dim = last_enc_dim * 2 * ((self.freq_dim // (2 ** len(enc_channels))) + 1)
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_size = rnn_hidden_size
        
        self.complex_lstm = ComplexLSTM(
            input_size=rnn_input_dim // 2,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=rnn_bidirectional
        )
        
        lstm_out_dim = rnn_hidden_size * (2 if rnn_bidirectional else 1)
        self.lstm_proj = nn.Linear(lstm_out_dim * 2, rnn_input_dim)
        
        self.decoders = nn.ModuleList()
        for i in range(len(enc_channels)):
            in_ch = enc_channels[-i-1] * 2
            out_ch = enc_channels[-i-2] * 2 if i < len(enc_channels)-1 else in_channels * 2
            self.decoders.append(
                self._complex_deconv_block(in_ch * 2, out_ch, kernel_size=(2, 5), stride=(1, 2))
            )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _complex_conv_block(self, in_ch, out_ch, kernel_size, stride):
        padding = [k//2 for k in kernel_size] if isinstance(kernel_size, tuple) else kernel_size // 2
        return nn.Sequential(
            ComplexConv2d(in_ch//2, out_ch//2, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if self.activation == 'relu' else nn.ELU(inplace=True)
        )
    
    def _complex_deconv_block(self, in_ch, out_ch, kernel_size, stride):
        padding = [k//2 for k in kernel_size] if isinstance(kernel_size, tuple) else kernel_size // 2
        return nn.Sequential(
            ComplexTransposeConv2d(in_ch//2, out_ch//2, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True) if self.activation == 'relu' else nn.ELU(inplace=True)
        )
    
    def forward(self, x):
        """
        Input: (B, C, F, T)  magnitude spectrogram
        Output: (B, C, F, T) enhanced magnitude spectrogram
        """
        _, _, F_in, T_in = x.shape
        x = x.permute(0, 1, 3, 2).contiguous() # (B, C, T, F)

        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, C * F)
        
        rnn_out = self.complex_lstm(x)
        rnn_out = self.lstm_proj(rnn_out)
        
        rnn_out = rnn_out.view(B, T, C, F)
        rnn_out = rnn_out.permute(0, 2, 1, 3).contiguous()  # (B, C, T, F)
        
        x = rnn_out
        for i, dec in enumerate(self.decoders):
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = dec(x)

        x = x.permute(0, 1, 3, 2).contiguous()
        
        mask = self.output_conv(x)
        
        return mask