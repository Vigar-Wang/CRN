import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, features, time_pooling, freq_pooling):
        super().__init__()
        assert len(features) == len(time_pooling) == len(freq_pooling)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool_t = time_pooling
        self.pool_f = freq_pooling

        # 编码器
        for i, ch in enumerate(features):
            if i == 0:
                self.encoder.append(self._conv_block(in_channels, ch))
            else:
                self.encoder.append(self._conv_block(features[i-1], ch))

        # 瓶颈层
        self.bottleneck = self._conv_block(features[-1], features[-1]*2)

        # 解码器
        for i, ch in enumerate(reversed(features)):
            if i == 0:
                self.decoder.append(self._conv_block(features[-1]*2 + ch, ch))
            else:
                self.decoder.append(self._conv_block(features[-i] + ch, ch))

        # 最终输出层
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

    def forward(self, x):
        # x: (B, C, F, T)
        skips = []
        # 编码
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=(self.pool_f[i], self.pool_t[i]))
        # 瓶颈
        x = self.bottleneck(x)
        # 解码
        skips = skips[::-1]
        for i, dec in enumerate(self.decoder):
            target_size = skips[i].shape[2:]
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            x = torch.cat([x, skips[i]], dim=1)
            x = dec(x)
        out = self.final_conv(x)
        return out