import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.fft import fft2, ifft2
import math
from einops import repeat
from mamba_ssm import selective_scan_fn  # 需要安装 mamba_ssm: pip install mamba_ssm

# SS2D 模块（用户提供的代码，稍作整理）
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

# 权重初始化
def initialize_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# 频域增强
def dct_enhance(x):
    x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    x_freq = fft2(x)
    x_freq = SS2D(d_model=x.shape[-1], d_state=8, d_conv=3, expand=1.5)(x_freq)
    x = ifft2(x_freq).real
    x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    return x

# 编码器：仅在 CNN 特征提取部分融合 SS2D
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        # 支持多光谱输入
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改 layer3 和 layer4 的步幅，保留更高分辨率
        for n, m in self.resnet.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.resnet.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # SS2D 增强 layer1 输出
        self.ss2d = SS2D(d_model=64, d_state=16, d_conv=3, expand=2)
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(128+256+512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        initialize_weights(self.conv_fuse)

    def forward(self, x):
        x0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))  # [B, 64, H/2, W/2]
        xm = self.resnet.maxpool(x0)
        x1 = self.resnet.layer1(xm)  # [B, 64, H/4, W/4]
        # SS2D 增强
        x1_perm = x1.permute(0, 2, 3, 1)  # [B, H/4, W/4, 64]
        x1_ss2d = self.ss2d(x1_perm)  # [B, H/4, W/4, 64]
        x1 = x1 + x1_ss2d.permute(0, 3, 1, 2)  # 残差连接
        x2 = self.resnet.layer2(x1)  # [B, 128, H/4, W/4]
        x3 = self.resnet.layer3(x2)  # [B, 256, H/4, W/4]
        x4 = self.resnet.layer4(x3)  # [B, 512, H/4, W/4]
        x4 = torch.cat([x2, x3, x4], dim=1)  # [B, 128+256+512, H/4, W/4]
        x4 = self.conv_fuse(x4)  # [B, 128, H/4, W/4]
        return x0, x1, x4

# 残差块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# 解码块
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.block = self._make_layers_(in_channels, out_channels)
        self.cb = ConvBlock(out_channels)

    def _make_layers_(self, in_channels, out_channels, blocks=2, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResBlock(in_channels, out_channels, stride, downsample)]
        for i in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        x = self.cb(x)
        return x

# 语义解码器
class LCMDecoder(nn.Module):
    def __init__(self):
        super(LCMDecoder, self).__init__()
        self.db4_1 = DecoderBlock(128+64, 128)
        self.db1_0 = DecoderBlock(128+64, 128)

    def decode(self, x1, x2, db):
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        x = db(x)
        return x

    def forward(self, x0, x1, x4):
        x1 = self.decode(x4, x1, self.db4_1)
        x0 = self.decode(x1, x0, self.db1_0)
        return x0, x1

# 变化检测分支
class CDBranch(nn.Module):
    def __init__(self):
        super(CDBranch, self).__init__()
        self.db4 = DecoderBlock(256+128, 128)
        self.db1 = DecoderBlock(256, 128)
        self.db0 = DecoderBlock(256, 128)

    def decode(self, x1, x2, db):
        x1 = db(x1)
        x1 = F.upsample(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x1, x2], 1)
        return x

    def forward(self, x0, x1, x1_4, x2_4):
        x4 = torch.cat([x1_4, x2_4, torch.abs(x1_4 - x2_4)], 1)
        x1 = self.decode(x4, x1, self.db4)
        x0 = self.decode(x1, x0, self.db1)
        x0 = self.db0(x0)
        return x0

# 主网络
class TBFFNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6):  # SECOND 数据集默认 6 类
        super(TBFFNet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.lcm_decoder = LCMDecoder()
        self.cd_branch = CDBranch()
        self.lcm_classifier1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.lcm_classifier2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        self.cd_classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        # 辅助分类器
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        initialize_weights(self.lcm_decoder, self.cd_branch, self.lcm_classifier1, self.lcm_classifier2, self.cd_classifier, self.aux_classifier)

    def forward(self, x1, x2):
        x_size = x1.size()
        # 频域增强
        x1 = dct_enhance(x1)
        x2 = dct_enhance(x2)
        # 编码
        x1_0, x1_1, x1_4 = self.encoder(x1)
        x2_0, x2_1, x2_4 = self.encoder(x2)
        # 解码
        x1_0, x1_1 = self.lcm_decoder(x1_0, x1_1, x1_4)
        x2_0, x2_1 = self.lcm_decoder(x2_0, x2_1, x2_4)
        # 变化检测
        cd_map = self.cd_branch(torch.abs(x1_0 - x2_0), torch.abs(x1_1 - x2_1), x1_4, x2_4)
        change = self.cd_classifier(cd_map)
        # 语义分割
        out1 = self.lcm_classifier1(x1_0)
        out2 = self.lcm_classifier2(x2_0)
        # 辅助输出
        aux_out1 = self.aux_classifier(x1_1)
        aux_out2 = self.aux_classifier(x2_1)
        return (
            F.upsample(change, x_size[2:], mode='bilinear'),
            F.upsample(out1, x_size[2:], mode='bilinear'),
            F.upsample(out2, x_size[2:], mode='bilinear'),
            F.upsample(aux_out1, x_size[2:], mode='bilinear'),
            F.upsample(aux_out2, x_size[2:], mode='bilinear')
        )

