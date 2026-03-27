import torch
import torchaudio
from util.audio_utils import STFT
from model import get_model

class BaseInferencer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model = self._load_model()
        self.stft = STFT(
            config['stft']['n_fft'],
            config['stft']['hop_length'],
            config['stft']['win_length'],
            config['stft']['window']
        )

    def _load_model(self):
        model_cfg = self.config['model']
        model_name = model_cfg['name']
        model_kwargs = model_cfg[model_name]
        model = get_model(model_name, **model_kwargs).to(self.device)
        checkpoint_path = self.config['inference']['checkpoint']  # 需要在配置中指定
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        # 处理可能的 _orig_mod. 前缀
        if all(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def enhance(self, noisy_wave):
        """输入: 波形张量 (T,) 或 (1, T)"""
        if noisy_wave.dim() == 1:
            noisy_wave = noisy_wave.unsqueeze(0)
        noisy_wave = noisy_wave.to(self.device)
        with torch.no_grad():
            noisy_spec = self.stft(noisy_wave)
            noisy_mag = torch.abs(noisy_spec).unsqueeze(1)
            pred_mag = self.model(noisy_mag)
            pred_spec = pred_mag.squeeze(1) * torch.exp(1j * torch.angle(noisy_spec))
            enhanced_wave = self.stft.inverse(pred_spec, length=noisy_wave.shape[-1])
        return enhanced_wave.cpu()