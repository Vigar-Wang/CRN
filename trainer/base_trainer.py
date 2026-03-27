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
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=(self.device.type == 'cuda')
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # 验证时 batch_size 可固定为 1
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=(self.device.type == 'cuda')
        )

        # 优化器与调度器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['training']['lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # 损失函数
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

        # STFT 工具（用于重建波形，仅在 si_snr 损失时使用）
        stft_cfg = config['stft']
        self.stft = STFT(
            stft_cfg['n_fft'],
            stft_cfg['hop_length'],
            stft_cfg['win_length'],
            stft_cfg['window']
        )

        # 日志与保存
        self.writer = SummaryWriter(log_dir=os.path.join(config['training']['checkpoint_dir'], 'logs'))
        self.checkpoint_dir = config['training']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 恢复训练或加载预训练权重
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
        torch.save(checkpoint, path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved (val loss {self.best_val_loss:.4f})")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, (noisy_wave, clean_wave) in enumerate(self.train_loader):
            noisy_wave = noisy_wave.to(self.device)
            clean_wave = clean_wave.to(self.device)

            noisy_spec = self.stft(noisy_wave)
            clean_spec = self.stft(clean_wave)

            noisy_mag = np.abs(noisy_spec)
            clean_mag = np.abs(clean_spec)

            self.optimizer.zero_grad()
            pred_mag = self.model(noisy_mag.unsqueeze(1))

            if self.config['training']['loss'] == 'mse':
                loss = self.criterion(pred_mag, clean_mag.unsqueeze(1))
            else:
                pred_spec = pred_mag.squeeze(1) * torch.exp(1j * torch.angle(noisy_spec))
                pred_wave = self.stft.inverse(pred_spec, length=clean_wave.shape[-1])
                if self.config['training']['loss'] == 'Mel':
                    loss = self.criterion(pred_wave, clean_wave, sr=self.config['data']['sample_rate'])
                elif self.config['training']['loss'] in {'si_snr', 'MR_STFT'}:
                    loss = self.criterion(pred_wave, clean_wave)
                else:
                    raise NotImplementedError("loss function unachieved")

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

                noisy_spec = self.stft(noisy_wave)
                clean_spec = self.stft(clean_wave)

                noisy_mag = np.abs(noisy_spec)
                clean_mag = np.abs(clean_spec)

                pred_mag = self.model(noisy_mag.unsqueeze(1))
                if self.config['training']['loss'] == 'mse':
                    loss = self.criterion(pred_mag, clean_mag.unsqueeze(1))
                else:
                    pred_spec = pred_mag.squeeze(1) * torch.exp(1j * torch.angle(noisy_spec))
                    pred_wave = self.stft.inverse(pred_spec, length=clean_wave.shape[-1])
                    if self.config['training']['loss'] == 'Mel':
                        loss = self.criterion(pred_wave, clean_wave, sr=self.config['data']['sample_rate'])
                    elif self.config['training']['loss'] in {'si_snr', 'MR_STFT'}:
                        loss = self.criterion(pred_wave, clean_wave)
                    else:
                        raise NotImplementedError("loss function unachieved")

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

            # 学习率记录
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('train/lr', lr, epoch)

        print("Training finished.")