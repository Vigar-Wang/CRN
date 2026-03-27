import torch
from util.audio_utils import mel_log

def si_snr_loss(est, target, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    dot = (est * target).sum(dim=-1, keepdim=True)
    target_norm = (target * target).sum(dim=-1, keepdim=True)
    s_target = dot / (target_norm + eps) * target
    e_noise = est - s_target
    snr = 10 * torch.log10((s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps)
    return -snr.mean()

def stft_loss(pred, target, fft_size=512, hop_length=128, win_length=512, window_fn='hann_window'):
    window = getattr(torch, window_fn)(win_length)
    pred_stft = torch.stft(pred, fft_size, hop_length, win_length, window=window, return_complex=True)
    target_stft = torch.stft(target, fft_size, hop_length, win_length, window=window, return_complex=True)
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    
    mag_loss = torch.nn.functional.l1_loss(pred_mag, target_mag)
    log_mag_loss = torch.nn.functional.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
    return mag_loss + log_mag_loss

def MR_STFT_loss(est, target, fft_sizes=[512, 1024, 2048], hop_lengths=[128, 256, 512],
        win_lengths=[512, 1024, 2048]):
    loss = 0
    for fft, hop, win in zip(fft_sizes, hop_lengths, win_lengths):
        loss += stft_loss(est, target, fft, hop, win)
    return loss / len(fft_sizes)

def Mel_loss(est, target, sr=8000):
    pred_mel = mel_log(est, sr)
    target_mel = mel_log(target, sr)
    return torch.nn.functional.l1_loss(pred_mel, target_mel)