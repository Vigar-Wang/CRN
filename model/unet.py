import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features, time_pooling, freq_pooling, decode_mode):
        super().__init__()
        assert len(features) == len(time_pooling) == len(freq_pooling)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.pool_t = time_pooling
        self.pool_f = freq_pooling
        self.decode_mode = decode_mode

        for i, ch in enumerate(features):
            if i == 0:
                self.encoder.append(self._conv_block(in_channels, ch))
            else:
                self.encoder.append(self._conv_block(features[i-1], ch))

        self.bottleneck = self._conv_block(features[-1], features[-1]*2)

        if self.decode_mode == 'interpolate':
            for i, ch in enumerate(reversed(features)):
                in_ch = features[-1]*2 + ch if i == 0 else features[-i] + ch
                out_ch = ch
                self.decoder.append(self._conv_block(in_ch, out_ch))
        elif self.decode_mode == 'transposed_conv':
            for i in range(len(features)):
                in_ch = features[-1]*2 if i == 0 else features[-i]
                out_ch = features[-(i+1)]
                self.up_convs.append(self._up_conv_block(in_ch, out_ch))
            for ch in reversed(features):
                self.decoder.append(self._conv_block(ch * 2, ch))
        else:
            raise ValueError(f"Unknown decoder: {self.decode_mode}")

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x):
        # x: (B, C, F, T)
        skips = []
        _, _, in_f_size, in_t_size = x.shape

        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(self.pool_f[i], self.pool_t[i]))

        x = self.bottleneck(x)

        skips = skips[::-1]
        for i, dec in enumerate(self.decoder):
            skip = skips[i]
            if self.decode_mode == 'interpolate':
                target_size = skips[i].shape[2:]
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            elif self.decode_mode == 'transposed_conv':
                x = self.up_convs[i](x)
                if x.shape[2] > skip.shape[2] or x.shape[3] > skip.shape[3]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                else:
                    _, _, h, w = skip.shape
                    target_h, target_w = x.shape[2], x.shape[3]
                    top = (h - target_h) // 2
                    left = (w - target_w) // 2
                    skip = skip[:, :, top:top+target_h, left:left+target_w]
            else:
                raise ValueError(f"Unknown decoder: {self.decode_mode}")
                
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        out = self.final_conv(x)

        if out.shape[2:] != (in_f_size, in_t_size):
            out = F.interpolate(out, size=(in_f_size, in_t_size), mode='bilinear', align_corners=False)

        return out