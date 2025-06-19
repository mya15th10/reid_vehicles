import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def weights_init_kaiming(m):
    """Weight Initialization using Kaiming method"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    """Initialize classifier weights"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class SIELayer(nn.Module):
    """Spatial-Instance Embedding Layer for camera-aware features"""
    
    def __init__(self, channels, camera_num=0, view_num=0, sie_xishu=1.0):
        super().__init__()
        self.sie_xishu = sie_xishu
        self.channels = channels
        self.camera_num = camera_num
        self.view_num = view_num
        
        if camera_num > 0:
            self.camera_embedding = nn.Parameter(torch.zeros(camera_num, channels))
            nn.init.normal_(self.camera_embedding, std=0.01)
        else:
            self.camera_embedding = None
            
        if view_num > 0:
            self.view_embedding = nn.Parameter(torch.zeros(view_num, channels))
            nn.init.normal_(self.view_embedding, std=0.01)
        else:
            self.view_embedding = None
    
    def forward(self, x, cam_label=None, view_label=None):
        """
        Args:
            x: feature tensor [batch_size, channels]
            cam_label: camera labels [batch_size]
            view_label: view labels [batch_size]
        """
        if self.camera_embedding is not None and cam_label is not None:
            # Ensure cam_label is within valid range
            cam_label = torch.clamp(cam_label, 0, self.camera_num - 1)
            x = x + self.sie_xishu * self.camera_embedding[cam_label]
            
        if self.view_embedding is not None and view_label is not None:
            # Ensure view_label is within valid range
            view_label = torch.clamp(view_label, 0, self.view_num - 1)
            x = x + self.sie_xishu * self.view_embedding[view_label]
            
        return x

class build_feature_transformer(nn.Module):
    """Fixed: Feature-based model for Vehicle Re-identification"""
    
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_feature_transformer, self).__init__()
        
        self.num_classes = num_classes
        self.cfg = cfg
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        
        # FIXED: Simple architecture for 256-dim features
        self.in_planes = 256
        self.embedding_dim = 256  # Keep same dimension
        
        print(f"Building feature-based model:")
        print(f"  - Input feature dim: {self.in_planes}")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Number of classes: {num_classes}")
        
        # Camera configuration
        self.camera_num = camera_num if cfg.MODEL.SIE_CAMERA else 0
        self.view_num = view_num if cfg.MODEL.SIE_VIEW else 0
        
        # FIXED: Simple processing - no complex layers
        self.feature_bn = nn.BatchNorm1d(self.in_planes)
        self.feature_bn.bias.requires_grad_(False)
        self.feature_bn.apply(weights_init_kaiming)
        
        # SIE Layer
        sie_coefficient = getattr(cfg.MODEL, 'SIE_COE', 3.0)
        self.sie_layer = SIELayer(
            channels=self.embedding_dim,
            camera_num=self.camera_num,
            view_num=self.view_num,
            sie_xishu=sie_coefficient
        )
        
        # Bottleneck
        self.bottleneck = nn.BatchNorm1d(self.embedding_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # Classifier
        self.classifier = nn.Linear(self.embedding_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # Input validation
        if len(x.shape) != 2 or x.shape[1] != self.in_planes:
            raise ValueError(f"Expected [batch_size, 256], got {x.shape}")
        
        # Simple processing
        global_feat = self.feature_bn(x)
        
        # Apply SIE
        if cam_label is not None:
            global_feat = self.sie_layer(global_feat, cam_label=cam_label, view_label=view_label)
        
        # Bottleneck
        feat = self.bottleneck(global_feat)
        
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            return F.normalize(feat, p=2, dim=1)
    def load_param(self, trained_path):
        """Load pretrained parameters"""
        try:
            param_dict = torch.load(trained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in param_dict:
                param_dict = param_dict['model']
            elif 'state_dict' in param_dict:
                param_dict = param_dict['state_dict']
            
            # Remove 'module.' prefix if present
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = key.replace('module.', '')
                new_param_dict[new_key] = value
            
            # Load parameters
            missing_keys, unexpected_keys = self.load_state_dict(new_param_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print(f'Successfully loaded pretrained model from {trained_path}')
            
        except Exception as e:
            print(f'Error loading pretrained model: {e}')
            raise e

def make_model(cfg, num_class, camera_num, view_num):
    """Create model based on configuration"""
    if cfg.MODEL.NAME == 'feature_transformer':  
        model = build_feature_transformer(num_class, camera_num, view_num, cfg)
        print('===========Building Feature-based Transformer===========')
    else:
        raise NotImplementedError(f"Model '{cfg.MODEL.NAME}' not implemented")
    
    return model