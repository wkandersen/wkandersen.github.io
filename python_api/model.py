import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import timm
from typing import Tuple

class TimmModel(LightningModule):
    def __init__(self, class_names: int, model_name: str = "resnet18", lr: float = 0.001, dropout: float = 0.2, num_features: int = None) -> None:
        super(TimmModel, self).__init__()
        # Load a pre-trained model from timm
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove the final classification layer
        backbone_output_features = self.backbone.num_features

        # Add custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone_output_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, class_names),
        )

        self.lr = lr

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        features = self.backbone(X)  # Extract features using the pre-trained model
        X = self.classifier(features)  # Pass features through the custom head
        return F.log_softmax(X, dim=1)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        X, y = train_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        X, y = val_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, test_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        X, y = test_batch
        y_hat = self(X)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).sum().item() / y.shape[0]
        self.log("test_loss", loss)
        self.log("test_acc", acc)
