import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def weights_init_kaiming(m):
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
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
class SIELayer(nn.Module):
    def __init__(self, channels, camera_num=0, view_num=0, sie_xishu=1.0):
        super().__init__()
        self.sie_xishu = sie_xishu
        self.channels = channels
        self.camera_num = camera_num
        self.view_num = view_num
        
        if camera_num > 0:
            self.camera_embedding = nn.Parameter(torch.zeros(camera_num, channels))
            nn.init.normal_(self.camera_embedding, std=0.01)
        if view_num > 0:
            self.view_embedding = nn.Parameter(torch.zeros(view_num, channels))
            nn.init.normal_(self.view_embedding, std=0.01)
    
    def forward(self, x, cam_label=None, view_label=None):
        B, C = x.shape
        if self.camera_num > 0 and cam_label is not None:
            x = x + self.sie_xishu * self.camera_embedding[cam_label]
        if self.view_num > 0 and view_label is not None:
            x = x + self.sie_xishu * self.view_embedding[view_label]
        return x
    


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        # x is in shape [seq_len, batch, dim]
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = residual + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim=768, depth=6, num_heads=8, mlp_ratio=4., drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
    def forward(self, x):
        # x is [batch, seq_len, dim]
        # Convert to [seq_len, batch, dim] for PyTorch's MultiheadAttention
        x = x.permute(1, 0, 2)
        
        for block in self.blocks:
            x = block(x)
            
        # Convert back to [batch, seq_len, dim]
        x = x.permute(1, 0, 2)
        return x

class VehicleTransformer(nn.Module):
    def __init__(self, num_classes, cnn_output_dim=2048, embedding_dim=768, 
             transformer_depth=6, num_heads=8, dropout=0.1, attn_dropout=0.1,
             sie_xishu=1.0, camera=0, view=0):
        super(VehicleTransformer, self).__init__()
        
        # Transformer Input Projection: CNN features -> Transformer dim
        self.input_proj = nn.Conv1d(cnn_output_dim, embedding_dim, kernel_size=1)
        # SIE
        self.sie = SIELayer(embedding_dim, camera_num=camera, view_num=view, sie_xishu=sie_xishu)
        # Position Embedding - learned position embeddings
        self.max_seq_len = 64  # Max number of CNN feature vectors (depends on your CNN output size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Transformer Encoder
        self.transformer = TransformerEncoder(
            dim=embedding_dim,
            depth=transformer_depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            drop_rate=dropout,
            attn_drop_rate=attn_dropout
        )
        
        # Bottleneck & Classifier
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, cam_label=None, view_label=None):
        batch_size = x.shape[0]
        
        # x shape [batch_size, cnn_output_dim, height, width]
        h, w = x.shape[2], x.shape[3]
        
        # Reshape CNN features to sequence of vectors
        x = x.view(batch_size, x.shape[1], -1)  # [batch_size, cnn_output_dim, h*w]
        
        # Project to transformer dimension
        x = self.input_proj(x)  # [batch_size, embedding_dim, h*w]
        x = x.permute(0, 2, 1)  # [batch_size, h*w, embedding_dim]
        
        # Truncate if sequence is too long
        seq_len = min(x.shape[1], self.max_seq_len - 1)  # -1 to make room for cls token
        x = x[:, :seq_len, :]
        
        # Add positional embedding
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Append class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer
        x = self.transformer(x)

        # Get class token for classification
        cls_token_out = x[:, 0]

        # Apply SIE
        cls_token_out = self.sie(cls_token_out, cam_label, view_label)
        
        # Apply bottleneck
        features = self.bottleneck(cls_token_out)
        
        if self.training:
            # For training, return logits and features
            logits = self.classifier(features)
            return features, logits
        else:
            # For inference, return normalized features
            features = F.normalize(features, p=2, dim=1)
            return features

class ResNetTransformer(nn.Module):
    def __init__(self, dim, depth=6, heads=8, mlp_dim=2048, dropout=0.0, 
                 attn_dropout=0.0, sie_xishu=1.0, camera=0, view=0):
        super(ResNetTransformer, self).__init__()
        self.in_planes = dim
        self.ID_LOSS_TYPE = 'softmax'  # Default loss type
        self.neck = 'bnneck'  # Default neck type
        self.neck_feat = 'after'  # Default neck feature option
        
        # CNN backbone sẽ được xử lý từ bên ngoài (từ self.base trong build_cnn_transformer)
        
        # Transformer for processing CNN features
        self.transformer = VehicleTransformer(
            num_classes=0,  # Số lượng classes sẽ được thiết lập từ bên ngoài
            cnn_output_dim=dim,
            embedding_dim=dim,
            transformer_depth=depth,
            num_heads=heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            sie_xishu=sie_xishu,
            camera=camera,
            view=view
        )
        
        # Bottleneck (sẽ được xử lý từ bên ngoài trong build_cnn_transformer)
        self.bottleneck = nn.BatchNorm1d(dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
    def forward(self, x, cam_label=None, view_label=None):
        # x là đặc trưng đã được trích xuất từ CNN
        # Sử dụng transformer
        global_feat = self.transformer(x, cam_label=cam_label, view_label=view_label)
        
        # Return đặc trưng đã qua SIE và transformer
        return global_feat



