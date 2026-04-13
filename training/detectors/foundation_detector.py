import logging
import math
import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train
from .base_detector import AbstractDetector

from transformers import SiglipVisionModel

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

class TemporalSafetyClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4, num_labels=1, dropout=0.1):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, frame_embeddings):
        x = self.pos_enc(frame_embeddings)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.head(x)  # (B, num_labels)
        return logits

@DETECTOR.register_module(module_name='foundation_detector')
class FoundationDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        
        # Load the pre-trained SigLIP backbone
        self.backbone = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')
        
        # Freeze the backbone 
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Initialize Temporal Classifier
        self.model = TemporalSafetyClassifier(
            embed_dim=self.backbone.config.hidden_size, 
            num_heads=8, 
            num_layers=4, 
            num_labels=1
        )
        
        # Standard BCE loss for binary classification
        self.loss_func = nn.BCELoss()

    def build_backbone(self, config):
        pass 

    def build_loss(self, config):
        pass

    def features(self, data_dict: dict) -> torch.tensor:
        # DeepfakeBench provides videos in data_dict['image'] with shape (B, T, C, H, W)
        videos = data_dict['image']
        B, T, C, H, W = videos.shape
        
        # Re-shape for the image-level backbone: (B*T, C, H, W)
        v_flat = videos.reshape(B * T, C, H, W)
        
        # Extract features from frozen backbone
        with torch.no_grad():
            emb = self.backbone(v_flat).pooler_output
            
        # Re-shape back to sequence format (B, T, Embed_Dim)
        emb = emb.view(B, T, -1)
        
        return emb

    def classifier(self, features: torch.tensor):
        # Process temporal features
        logits = self.model(features)
        
        # Apply sigmoid for probability output (maintain B, 1 shape)
        prob = torch.sigmoid(logits)
        
        return prob

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].float().unsqueeze(1)
        pred = pred_dict['cls']
        
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        prob = self.classifier(features)
        
        pred_dict = {'cls': prob, 'prob': prob, 'feat': features}
        return pred_dict
