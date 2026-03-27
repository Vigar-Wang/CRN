import argparse
import yaml
from data import get_dataset
from model import get_model
from trainer.base_trainer import BaseTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.safe_load(f)

    # 获取数据集参数
    data_cfg = config['data']
    dataset_name = data_cfg.pop('dataset_name')   # 移除 dataset_name，剩下的作为 kwargs
    # 注意：train_txt 和 val_txt 需要分别传给训练和验证数据集
    train_kwargs = data_cfg.copy()
    train_kwargs['txt_path'] = data_cfg['train_txt']
    train_kwargs['limit'] = data_cfg['train_limit']
    val_kwargs = data_cfg.copy()
    val_kwargs['txt_path'] = data_cfg['val_txt']
    val_kwargs['limit'] = data_cfg['val_limit']

    # 创建数据集
    train_dataset = get_dataset(dataset_name, **train_kwargs)
    val_dataset = get_dataset(dataset_name, **val_kwargs)

    # 创建模型
    model_cfg = config['model']
    model_name = model_cfg['name']
    model_kwargs = model_cfg[model_name]
    model = get_model(model_name, **model_kwargs)

    # 训练器
    trainer = BaseTrainer(config, model, train_dataset, val_dataset)
    trainer.train()

if __name__ == '__main__':
    main()