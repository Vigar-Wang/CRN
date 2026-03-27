import torch
import torchaudio
from . import register_dataset

@register_dataset('DatasetMixed')
class DatasetMixed(torch.utils.data.Dataset):
    def __init__(self, train_txt=None, val_txt=None, sample_rate=16000, segment_seconds=None,
                segment_mode='random_crop', **kwargs):
        txt_path = kwargs.get('txt_path')
        if txt_path is None:
            raise ValueError("txt_path must be provided")
        self.txt_path = txt_path
        self.sample_rate = sample_rate
        self.segment_len = int(segment_seconds * sample_rate) if segment_seconds else None
        self.segment_mode = segment_mode

        # 读取文件对列表
        limit = kwargs.get('limit')
        data_list = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    clean_path, noisy_path = parts[0], parts[1]
                    data_list.append((clean_path, noisy_path))
        if limit:
            data_list = data_list[:limit]
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.data_list[idx]
        clean_wave, sr_clean = torchaudio.load(clean_path)
        noisy_wave, sr_noisy = torchaudio.load(noisy_path)

        # 重采样
        if sr_clean != self.sample_rate:
            clean_wave = torchaudio.functional.resample(clean_wave, sr_clean, self.sample_rate)
        if sr_noisy != self.sample_rate:
            noisy_wave = torchaudio.functional.resample(noisy_wave, sr_noisy, self.sample_rate)

        # 转为单声道
        if clean_wave.shape[0] > 1:
            clean_wave = clean_wave.mean(dim=0, keepdim=True)
        if noisy_wave.shape[0] > 1:
            noisy_wave = noisy_wave.mean(dim=0, keepdim=True)

        clean_wave = clean_wave.squeeze(0)
        noisy_wave = noisy_wave.squeeze(0)

        # 长度处理（训练模式）
        if self.segment_len is not None:
            if self.segment_mode == 'random_crop':
                if clean_wave.shape[0] >= self.segment_len:
                    start = torch.randint(0, clean_wave.shape[0] - self.segment_len + 1, (1,)).item()
                    clean_wave = clean_wave[start:start+self.segment_len]
                    noisy_wave = noisy_wave[start:start+self.segment_len]
                else:
                    clean_wave = self._pad_to_len(clean_wave, self.segment_len)
                    noisy_wave = self._pad_to_len(noisy_wave, self.segment_len)
            elif self.segment_mode == 'pad':
                clean_wave = self._pad_to_len(clean_wave, self.segment_len)
                noisy_wave = self._pad_to_len(noisy_wave, self.segment_len)

        return noisy_wave, clean_wave

    def _pad_to_len(self, wave, target_len):
        if wave.shape[0] >= target_len:
            return wave[:target_len]
        else:
            pad_len = target_len - wave.shape[0]
            return torch.nn.functional.pad(wave, (0, pad_len))