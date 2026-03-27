import torch
import torchaudio

class STFT:
    def __init__(self, n_fft, hop_length, win_length, window_fn='hann'):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = getattr(torch, window_fn)(win_length)

    def __call__(self, waveform):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(waveform.device),
            return_complex=True
        )
        return spec

    def inverse(self, spec, length=None):
        waveform = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(spec.device),
            length=length
        )
        return waveform

def si_snr_loss(est, target, eps=1e-8):
    """尺度不变信噪比损失 (负的 SI-SNR)"""
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    dot = (est * target).sum(dim=-1, keepdim=True)
    target_norm = (target * target).sum(dim=-1, keepdim=True)
    s_target = dot / (target_norm + eps) * target
    e_noise = est - s_target
    snr = 10 * torch.log10((s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps)
    return -snr.mean()