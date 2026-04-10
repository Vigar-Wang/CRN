import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.audio_utils import STFT
from util.loss_functions import *

class BaseTrainer:
    def __init__(self, config, model, train_dataset, val_dataset):
        self.config = config
        self.device = torch.device(config['training']['device'])
        self.model = model.to(self.device)
        self.model_type = config['training']['model_type']
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=(self.device.type == 'cuda')
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=(self.device.type == 'cuda')
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        if config['training']['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        elif config['training']['loss'] == 'si_snr':
            self.criterion = si_snr_loss
        elif config['training']['loss'] == 'MR_STFT':
            self.criterion = MR_STFT_loss
        elif config['training']['loss'] == 'Mel':
            self.criterion = Mel_loss
        else:
            raise ValueError(f"Unknown loss: {config['training']['loss']}")

        stft_cfg = config['stft']
        self.stft = STFT(
            stft_cfg['n_fft'],
            stft_cfg['hop_length'],
            stft_cfg['win_length'],
            stft_cfg['window']
        )

        self.writer = SummaryWriter(log_dir=os.path.join(config['training']['checkpoint_dir'], 'logs'))
        self.checkpoint_dir = config['training']['checkpoint_dir']
        self.save_interval = config['training']['save_interval']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.start_epoch = 1
        self.best_val_loss = float('inf')
        if config['training'].get('resume'):
            self._load_checkpoint(config['training']['resume'], resume_optimizer=True)
        elif config['training'].get('preload'):
            self._load_checkpoint(config['training']['preload'], resume_optimizer=False)

    def _load_checkpoint(self, path, resume_optimizer=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"Resumed from {path}, epoch {checkpoint['epoch']}, best val loss {self.best_val_loss:.4f}")
        else:
            print(f"Loaded pre-trained weights from {path}")

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
        if epoch % self.save_interval == 0:
            torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved (val loss {self.best_val_loss:.4f})")

    def model_input_get(self, spec):
        if self.model_type == 'CED':
            noisy_mag = torch.abs(spec)
            noisy_mag = noisy_mag.unsqueeze(1)
            return noisy_mag
        elif self.model_type == 'IRM':
            raise NotImplementedError("Model type not implemented")
        elif self.model_type == 'IPM':
            raise NotImplementedError("Model type not implemented")
        elif self.model_type == 'CRM':
            noisy_spec = torch.stack([spec.real, spec.imag], dim=1)
            # noisy_spec = noisy_spec.permute(0, 1, 3, 2).contiguous()
            return noisy_spec
        else:
            raise NotImplementedError("Model type not implemented")

    def model_output_post_process(self, model_output, origin_spec=0, length=0, type='mag'):
        if self.model_type == 'CED':
            if type == 'mag':
                return model_output
            else:
                pred_spec = model_output.squeeze(1) * torch.exp(1j * torch.angle(spec))
                pred_wave = self.stft.inverse(pred_spec, length=length)
                return pred_wave
        elif self.model_type == 'IRM':
            raise NotImplementedError("Model type not implemented")
        elif self.model_type == 'IPM':
            raise NotImplementedError("Model type not implemented")
        elif self.model_type == 'CRM':
            Y_real, Y_imag = origin_spec.real, origin_spec.imag
            M_real, M_imag = model_output[:, 0], model_output[:, 1]
            S_real = Y_real * M_real - Y_imag * M_imag
            S_imag = Y_real * M_imag + Y_imag * M_real
            S_hat = torch.complex(S_real, S_imag)
            # S_hat = S_hat.permute(0, 2, 3, 1).contiguous()
            if type == 'mag':
                return torch.abs(S_hat).unsqueeze(1)
            else:
                pred_wave = self.stft.inverse(S_hat, length=length)
                return pred_wave
        else:
            raise NotImplementedError("Model type not implemented")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (noisy_wave, clean_wave) in enumerate(self.train_loader):
            noisy_wave = noisy_wave.to(self.device)
            clean_wave = clean_wave.to(self.device)

            clean_spec = self.stft(clean_wave)
            clean_mag = torch.abs(clean_spec)

            noisy_spec = self.stft(noisy_wave)

            model_input_sig = self.model_input_get(noisy_spec)

            self.optimizer.zero_grad()
            pred_sig = self.model(model_input_sig)

            if self.config['training']['loss'] == 'mse':
                pred_mag = self.model_output_post_process(pred_sig, type='mag')
                loss = self.criterion(pred_mag, clean_mag.unsqueeze(1))
            else:
                pred_wave = self.model_output_post_process(pred_sig, origin_spec=noisy_spec, 
                    length=clean_wave.shape[-1], type='wave')
                if self.config['training']['loss'] == 'Mel':
                    loss = self.criterion(pred_wave, clean_wave, sr=self.config['data']['sample_rate'])
                elif self.config['training']['loss'] in {'si_snr', 'MR_STFT'}:
                    loss = self.criterion(pred_wave, clean_wave)
                else:
                    raise NotImplementedError("loss function not implemented")

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if batch_idx % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), epoch * len(self.train_loader) + batch_idx)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        self.writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for noisy_wave, clean_wave in self.val_loader:
                noisy_wave = noisy_wave.to(self.device)
                clean_wave = clean_wave.to(self.device)

                clean_spec = self.stft(clean_wave)
                clean_mag = torch.abs(clean_spec)

                noisy_spec = self.stft(noisy_wave)

                model_input_sig = self.model_input_get(noisy_spec)

                self.optimizer.zero_grad()
                pred_sig = self.model(model_input_sig)

                if self.config['training']['loss'] == 'mse':
                    pred_mag = self.model_output_post_process(model_output, type='mag')
                    loss = self.criterion(pred_mag, clean_mag.unsqueeze(1))
                else:
                    pred_wave = self.model_output_post_process(pred_sig, origin_spec=noisy_spec, 
                        length=clean_wave.shape[-1], type='wave')
                    if self.config['training']['loss'] == 'Mel':
                        loss = self.criterion(pred_wave, clean_wave, sr=self.config['data']['sample_rate'])
                    elif self.config['training']['loss'] in {'si_snr', 'MR_STFT'}:
                        loss = self.criterion(pred_wave, clean_wave)
                    else:
                        raise NotImplementedError("loss function not implemented")

                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch} | Val Loss: {avg_loss:.4f}")
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config['training']['epochs'] + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self._save_checkpoint(epoch, is_best)

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', lr, epoch)

        print("Training finished.")