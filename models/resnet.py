import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

# IBN block - kết hợp Instance Normalization và Batch Normalization
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

# Bottleneck block cho ResNet với IBN
class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ResNet với IBN
class ResNet_IBN(nn.Module):
    def __init__(self, last_stride, block, layers, ibn_cfg=('a', 'a', 'a', None)):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride, ibn=ibn_cfg[3])
        
        # Khởi tạo trọng số
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

# Hàm tạo ResNet50-IBN-a
def resnet50_ibn_a(last_stride=1, pretrained=False):
    """Tạo model ResNet-50-IBN-a
    Args:
        last_stride: Stride của layer cuối cùng
        pretrained: Sử dụng pretrained weights từ ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3], 
                       ibn_cfg=('a', 'a', 'a', None))
    
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        # Xóa bỏ các layer fc không sử dụng
        if 'fc.weight' in state_dict:
            state_dict.pop('fc.weight')
        if 'fc.bias' in state_dict:
            state_dict.pop('fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys when loading pretrained weights: {res.missing_keys}")
        print(f"Unexpected keys when loading pretrained weights: {res.unexpected_keys}")
    
    return model

# Class chính sử dụng cho nhận diện phương tiện
class VehicleReIDBackbone(nn.Module):
    def __init__(self, num_classes=576, embedding_dim=512, pretrained=True, last_stride=1):
        """
        Backbone CNN for model vehicles re-identification 
        
        Param
            num_classes (int): Số lượng phương tiện cần phân loại (576 cho tập train)
            embedding_dim (int): Kích thước vector đặc trưng
            pretrained (bool): Sử dụng pretrained weights từ ImageNet
            last_stride (int): Stride của layer cuối của ResNet
        """
        super(VehicleReIDBackbone, self).__init__()

        # Sử dụng ResNet50-IBN-a làm backbone
        self.backbone = resnet50_ibn_a(last_stride=last_stride, pretrained=pretrained)
        self.backbone_output_dim = 2048  # ResNet50's output dimension
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding layer 
        self.embedding_layer = nn.Linear(self.backbone_output_dim, embedding_dim)
        
        # Batch normalization để ổn định embedding
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.bn.bias.requires_grad_(False)  # Không sử dụng bias
        
        # Classification layer
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass qua mô hình
        
        Tham số:
            x (tensor): Batch ảnh đầu vào, shape [batch_size, 3, height, width]
            
        Trả về:
            features (tensor): Vector đặc trưng đã normalize, shape [batch_size, embedding_dim]
            logits (tensor): Logits cho classification, shape [batch_size, num_classes]
        """
        # Trích xuất đặc trưng qua backbone
        x = self.backbone(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # [batch_size, 2048]
        
        # Embedding
        embedding = self.embedding_layer(x)  # [batch_size, embedding_dim]
        
        # Batch normalization
        embedding_bn = self.bn(embedding)
        
        # Dropout cho phân loại
        embedding_dropout = self.dropout(embedding_bn)
        
        # Logits cho classification
        logits = self.classifier(embedding_dropout)
        
        # L2 normalization cho embedding
        features = F.normalize(embedding_bn, p=2, dim=1)
        
        return features, logits