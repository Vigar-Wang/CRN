import argparse
import yaml
import torchaudio
from inferencer.base_inferencer import BaseInferencer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='enhanced.wav')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)

    # 添加推理特定配置（如 checkpoint 路径）
    if 'inference' not in config:
        config['inference'] = {}
    config['inference']['checkpoint'] = input("Enter checkpoint path: ")  # 或从命令行传入

    inferencer = BaseInferencer(config)
    noisy_wave, sr = torchaudio.load(args.input)
    # 重采样到目标采样率（若需要）
    if sr != config['data']['sample_rate']:
        noisy_wave = torchaudio.functional.resample(noisy_wave, sr, config['data']['sample_rate'])
    enhanced = inferencer.enhance(noisy_wave)
    torchaudio.save(args.output, enhanced, config['data']['sample_rate'])
    print(f"Enhanced audio saved to {args.output}")

if __name__ == '__main__':
    main()