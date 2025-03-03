import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import os
import copy
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict, deque
import time
import datetime
import math
# 环境检查函数
def check_environment():
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前设备: {torch.cuda.get_device_name(0)}")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


# 数据增强类（RGB处理）
class OCTBYOLTransform:
    """适用于OCT图像的BYOL增强策略"""

    def __init__(self, img_size=224):
        # 视图1（强增强）
        self.view1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply([transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 视图2（弱增强）
        self.view2 = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.view1(x), self.view2(x)


'''
替换国内镜像
'''
import torch.hub
_original_load_state_dict_from_url = torch.hub.load_state_dict_from_url # 保存原始的下载函数
def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
    url = url.replace("download.pytorch.org", "pytorch.tuna.tsinghua.edu.cn") # 将官方地址替换为清华镜
    return _original_load_state_dict_from_url(url, model_dir, map_location, progress, check_hash, file_name)
torch.hub.load_state_dict_from_url = load_state_dict_from_url

# 标准ResNet50编码器
class ResNet50Encoder(nn.Module):
    """保持原始ResNet50结构的编码器"""

    def __init__(self, pretrained=True):
        super().__init__()
        
        original = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        # 保留所有原始层
        self.conv1 = original.conv1  # 输入层 [3,224,224] -> [64,112,112]
        self.bn1 = original.bn1
        self.relu = original.relu
        self.maxpool = original.maxpool  # [64,112,112] -> [64,56,56]
        self.layer1 = original.layer1  # 残差块组1
        self.layer2 = original.layer2  # 残差块组2
        self.layer3 = original.layer3  # 残差块组3
        self.layer4 = original.layer4  # 残差块组4
        self.avgpool = original.avgpool  # 全局平均池化

        # 移除原始分类头
        del original.fc

    def forward(self, x):
        # 标准前向传播流程
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x.flatten(1)  # 输出形状: [batch_size, 2048]


# 投影头
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# BYOL主模型
class BYOL_OCT(nn.Module):
    def __init__(self, feature_dim=2048, tau=0.996):
        super().__init__()
        self.tau = tau

        # 在线网络
        self.online_encoder = ResNet50Encoder()
        self.online_projector = ProjectionHead(feature_dim)
        self.online_predictor = ProjectionHead(256, 512, 256)

        # 目标网络（深拷贝初始化）
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # 冻结目标网络
        self._freeze_target()

    def _freeze_target(self):
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        # 动量更新参数
        for o_param, t_param in zip(self.online_encoder.parameters(),
                                    self.target_encoder.parameters()):
            t_param.data = t_param.data * self.tau + o_param.data * (1 - self.tau)

        for o_param, t_param in zip(self.online_projector.parameters(),
                                    self.target_projector.parameters()):
            t_param.data = t_param.data * self.tau + o_param.data * (1 - self.tau)

    def forward(self, view1, view2):
        # 在线特征
        online_z1 = self.online_projector(self.online_encoder(view1))
        online_q1 = self.online_predictor(online_z1)
        online_z2 = self.online_projector(self.online_encoder(view2))
        online_q2 = self.online_predictor(online_z2)

        # 目标特征
        with torch.no_grad():
            target_z1 = F.normalize(self.target_projector(self.target_encoder(view1)), dim=-1)
            target_z2 = F.normalize(self.target_projector(self.target_encoder(view2)), dim=-1)

        # 对称损失
        loss = 2 - (F.cosine_similarity(online_q1, target_z2, dim=-1).mean() +
                    F.cosine_similarity(online_q2, target_z1, dim=-1).mean())
        return loss


def train_epoch(model, dataloader, epoch, optimizer, scaler, device, log_writer):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('tau', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    for data_iter_step, ((view1, view2), _) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        view1, view2 = view1.to(device), view2.to(device)

        optimizer.zero_grad()
        with autocast():
            loss = model(view1, view2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_target()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        metric_logger.update(tau=model.tau)

        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(dataloader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('tau', model.tau, epoch_1000x)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(epoch, model, optimizer, path):
    state = {
        'epoch': epoch,
        'model': model.online_encoder.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def main():
    # 环境检查
    check_environment()

    # 数据准备
    dataset = datasets.ImageFolder(
        root=r"/mnt/d/3.dlProject/bdrv/data/pretraindata(crop)",
        transform=OCTBYOLTransform()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )


    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BYOL_OCT().to(device)

    # 动态EMA动量配置
    initial_tau = 0.996
    final_tau = 0.999
    model.tau = initial_tau

    # 动态EMA动量更新函数
    def update_ema_momentum(epoch):
        progress = epoch / total_epochs
        model.tau = final_tau - (final_tau - initial_tau) * (1 + math.cos(math.pi * progress)) / 2

    # 优化器配置
    base_lr = 4e-4
    # 增强权重衰减（对抗小数据过拟合）
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    # 学习率调度器
    total_epochs = 250
    warmup_epochs = 15
    cosine_t0 = 30

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=0.01,  # 从base_lr的1%开始
                end_factor=1.0,
                total_iters=warmup_epochs
            ),
            CosineAnnealingWarmRestarts(
                optimizer,
                T_0=cosine_t0,
                T_mult=1,  # 固定周期长度
                eta_min=5e-4 * 0.01  # 最小学习率为初始值的1%
            )
        ],
        milestones=[warmup_epochs]
    )
    scaler = GradScaler()

    # 训练准备
    save_dir = "/mnt/d/3.dlProject/keyan/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    # 建立训练日志
    log_writer = SummaryWriter(log_dir=save_dir)
    # 训练循环
    start_time = time.time()
    for epoch in range(1, total_epochs + 1):
        update_ema_momentum(epoch)
        train_stats = train_epoch(model, dataloader, epoch, optimizer, scaler, device, log_writer)
        scheduler.step()
        # 保存检查点
        if epoch % 50 == 0 or epoch == total_epochs:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch}.pth")
            save_checkpoint(epoch, model, optimizer, checkpoint_path)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(save_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()