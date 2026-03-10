import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import SqueezeNet1_0_Weights, DenseNet121_Weights, DenseNet201_Weights

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel Attention
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).view(x.size(0), x.size(1), 1, 1)
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca

        # 2. Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sa

class PretrainedCatDetector(nn.Module):
    def __init__(self, num_classes, num_anchors=1):
        super(PretrainedCatDetector, self).__init__()

        # Backbone: DenseNet121
        weights = DenseNet121_Weights.DEFAULT
        densenet = models.densenet121(weights=weights)
        self.backbone = densenet.features

        # --- เพิ่ม Attention ตรงนี้ ---
        # DenseNet121 features ออกมาเป็น 1024 channels
        self.attention = CBAM(channels=1024)
        # ---------------------------

        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.prediction_per_anchor = 5 + num_classes

        self.det_head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_anchors * self.prediction_per_anchor, 1)
        )

    def forward(self, x):
        features = self.backbone(x) # [B, 1024, 7, 7]
        features = self.attention(features)

        det_raw = self.det_head(features)
        B, _, Hg, Wg = det_raw.shape
        det_out = det_raw.view(B, self.num_anchors, self.prediction_per_anchor, Hg, Wg)
        det_out = det_out.permute(0, 1, 3, 4, 2).contiguous()
        return det_out