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


class FeatureProcessor(nn.Module):
    """Process R-CNN features before classification"""
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        for m in self.processor:
            if isinstance(m, nn.Linear):
                weights_init_kaiming(m)
            elif isinstance(m, nn.BatchNorm1d):
                weights_init_kaiming(m)
    
    def forward(self, x):
        return self.processor(x)


class build_feature_transformer(nn.Module):
    """Feature-based Transformer model for Vehicle Re-identification
    
    This model processes R-CNN extracted features instead of raw images.
    Architecture: R-CNN Features -> Feature Processing -> SIE -> Bottleneck -> Classifier
    """
    
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_feature_transformer, self).__init__()
        
        # Configuration
        self.num_classes = num_classes
        self.cfg = cfg
        
        # Model settings
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        
        # Input feature dimension (R-CNN features)
        self.in_planes = getattr(cfg.MODEL, 'FEATURE_DIM', 256)  # Default to 256 if not specified
        self.embedding_dim = self.in_planes  # Keep same dimension for simplicity
        
        print(f"Building feature-based model:")
        print(f"  - Input feature dim: {self.in_planes}")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Camera num: {camera_num}")
        print(f"  - View num: {view_num}")
        
        # Camera and view configuration for SIE
        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
            
        if cfg.MODEL.SIE_VIEW:
            self.view_num = view_num
        else:
            self.view_num = 0
            
        # Feature input normalization
        self.feature_bn = nn.BatchNorm1d(self.in_planes)
        self.feature_bn.bias.requires_grad_(False)
        self.feature_bn.apply(weights_init_kaiming)
        
        # Feature processing layers
        dropout_rate = getattr(cfg.MODEL, 'DROP_OUT', 0.1)
        self.feature_processor = FeatureProcessor(
            input_dim=self.in_planes,
            output_dim=self.embedding_dim,
            dropout=dropout_rate
        )
        
        # SIE Layer for camera-aware features
        sie_coefficient = getattr(cfg.MODEL, 'SIE_COE', 3.0)
        self.sie_layer = SIELayer(
            channels=self.embedding_dim,
            camera_num=self.camera_num,
            view_num=self.view_num,
            sie_xishu=sie_coefficient
        )
        
        # Bottleneck layer
        self.bottleneck = nn.BatchNorm1d(self.embedding_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # Classifier based on loss type
        if self.ID_LOSS_TYPE == 'arcface':
            print('Using ArcFace loss with s:{}, m:{}'.format(
                cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(
                self.embedding_dim, 
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE, 
                m=cfg.SOLVER.COSINE_MARGIN
            )
        elif self.ID_LOSS_TYPE == 'cosface':
            print('Using CosFace loss with s:{}, m:{}'.format(
                cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(
                self.embedding_dim, 
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE, 
                m=cfg.SOLVER.COSINE_MARGIN
            )
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('Using AMSoftmax loss with s:{}, m:{}'.format(
                cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(
                self.embedding_dim, 
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE, 
                m=cfg.SOLVER.COSINE_MARGIN
            )
        elif self.ID_LOSS_TYPE == 'circle':
            print('Using Circle loss with s:{}, m:{}'.format(
                cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(
                self.embedding_dim, 
                self.num_classes,
                s=cfg.SOLVER.COSINE_SCALE, 
                m=cfg.SOLVER.COSINE_MARGIN
            )
        else:
            print('Using Softmax loss')
            self.classifier = nn.Linear(self.embedding_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Forward pass
        
        Args:
            x: R-CNN features [batch_size, feature_dim]
            label: ground truth labels for training [batch_size]
            cam_label: camera labels [batch_size] 
            view_label: view labels [batch_size]
            
        Returns:
            Training: (cls_score, global_feat)
            Testing: feat (after bottleneck) or global_feat (before bottleneck)
        """
        # Input validation
        if len(x.shape) != 2:
            raise ValueError(f"Expected 2D input [batch_size, feature_dim], got {x.shape}")
        
        if x.shape[1] != self.in_planes:
            raise ValueError(f"Expected feature dim {self.in_planes}, got {x.shape[1]}")
        
        # Feature normalization
        global_feat = self.feature_bn(x)
        
        # Feature processing
        global_feat = self.feature_processor(global_feat)
        
        # Apply SIE (camera-aware features)
        if cam_label is not None or view_label is not None:
            global_feat = self.sie_layer(global_feat, cam_label=cam_label, view_label=view_label)
        
        # Bottleneck
        feat = self.bottleneck(global_feat)
        
        # Training vs inference
        if self.training:
            # Classification score
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                if label is None:
                    raise ValueError("Label is required for metric learning losses during training")
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            
            return cls_score, global_feat  # Return both for combined losses
        else:
            # Inference mode
            if self.neck_feat == 'after':
                # Return features after bottleneck (normalized)
                return F.normalize(feat, p=2, dim=1)
            else:
                # Return features before bottleneck
                return F.normalize(global_feat, p=2, dim=1)
    
    def load_param(self, trained_path):
        """Load pretrained parameters"""
        try:
            param_dict = torch.load(trained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model' in param_dict:
                param_dict = param_dict['model']
            elif 'state_dict' in param_dict:
                param_dict = param_dict['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
            new_param_dict = {}
            for key, value in param_dict.items():
                new_key = key.replace('module.', '')
                new_param_dict[new_key] = value
            
            # Load parameters (strict=False to allow partial loading)
            missing_keys, unexpected_keys = self.load_state_dict(new_param_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys when loading pretrained model: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys when loading pretrained model: {unexpected_keys}")
                
            print(f'Successfully loaded pretrained model from {trained_path}')
            
        except Exception as e:
            print(f'Error loading pretrained model from {trained_path}: {e}')
            raise e
    
    def get_feature_dim(self):
        """Get output feature dimension"""
        return self.embedding_dim
    
    def get_classifier_weight(self):
        """Get classifier weights for analysis"""
        if hasattr(self.classifier, 'weight'):
            return self.classifier.weight
        else:
            return None

def make_model(cfg, num_class, camera_num, view_num):
    """Create model based on configuration
    
    Args:
        cfg: Configuration object
        num_class: Number of classes for classification
        camera_num: Number of cameras for SIE
        view_num: Number of views for SIE
        
    Returns:
        model: Built model ready for training/inference
    """
    if cfg.MODEL.NAME == 'feature_transformer':  
        model = build_feature_transformer(num_class, camera_num, view_num, cfg)
        print('===========Building Feature-based Transformer===========')
    elif cfg.MODEL.NAME == 'cnn_transformer':  # Your old config might use this
        model = build_feature_transformer(num_class, camera_num, view_num, cfg)
        print('===========Building Feature-based Transformer (CNN name)============')
    else:
        # Fallback for other model types
        raise NotImplementedError(f"Model '{cfg.MODEL.NAME}' not implemented. Available: 'feature_transformer'")
    
    return model