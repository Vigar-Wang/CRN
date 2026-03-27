import torch
import torchaudio
import torchaudio.transforms

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

def mel_log(audio, sr, n_fft=512, pre_enph_coef=0.97, hop_ms=0.01, \
    win_ms=0.025, power=2, mel_filter_num=26, mel_coef_num = 13):

    hop_point = round(sr*hop_ms)
    win_point = round(sr*win_ms)

    shifted = torch.cat([torch.zeros_like(audio[..., :1]), audio[..., :-1]], dim=-1)
    y_preemphasized = audio - pre_enph_coef*shifted

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, 
        n_fft=n_fft, 
        hop_length=hop_point, 
        win_length = win_point,
        n_mels=mel_filter_num, 
        window_fn=torch.hann_window,
        power=power
    )(y_preemphasized)

    log_mel = torchaudio.transforms.AmplitudeToDB(stype='power',
        top_db=80.0
    )(mel_spec)

    return log_mel
