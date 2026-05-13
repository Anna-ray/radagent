"""
radagent.federated.client
-------------------------
Hospital node for federated learning.

Author: Rayane Aggoune
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from radagent.federated.server import ClientUpdate


class HospitalNode:
    """A hospital node that trains locally and sends updates to the server.
    
    Args:
        node_id: Unique identifier for this hospital
        train_loader: DataLoader for local training data
        val_loader: DataLoader for local validation data
        device: Device to train on
        local_epochs: Number of local epochs per round
        lr: Learning rate for local training
    """
    
    def __init__(
        self,
        node_id: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        local_epochs: int = 1,
        lr: float = 1e-4,
    ):
        self.node_id = node_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        
    def local_train(
        self,
        model: nn.Module,
        round_number: int,
    ) -> ClientUpdate:
        """Train the model locally and return an update.
        
        Args:
            model: Global model to train
            round_number: Current federation round
            
        Returns:
            ClientUpdate with trained weights and metrics
        """
        start_time = time.time()
        
        # Move model to device
        model = model.to(self.device)
        model.train()
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        
        # Loss function: BCE with label masking
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Training loop
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(
                self.train_loader,
                desc=f"[{self.node_id}] Round {round_number}, Epoch {epoch+1}/{self.local_epochs}",
                leave=False,
            )
            
            for batch in pbar:
                images = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)
                label_mask = batch["label_mask"].to(self.device)
                
                # Forward pass
                logits = model(images)
                
                # Compute loss with masking
                loss_per_class = criterion(logits, labels)
                masked_loss = loss_per_class * label_mask
                
                # Average over present classes only
                loss = masked_loss.sum() / (label_mask.sum() + 1e-8)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = epoch_loss / num_batches
            print(f"[{self.node_id}] Round {round_number}, Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        # Evaluate on local validation set
        local_auc = self._evaluate(model)
        
        # Compute training time
        wall_clock_seconds = time.time() - start_time
        
        # Clone state dict to CPU (never send GPU tensors)
        state_dict = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        # Count samples
        num_samples = len(self.train_loader.dataset)
        
        return ClientUpdate(
            node_id=self.node_id,
            state_dict=state_dict,
            num_samples=num_samples,
            local_auc=local_auc,
            round_number=round_number,
            wall_clock_seconds=wall_clock_seconds,
        )
    
    def _evaluate(self, model: nn.Module) -> float:
        """Evaluate model on local validation set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Macro AUC across all classes
        """
        model.eval()
        
        all_labels = []
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["labels"].cpu().numpy()
                label_mask = batch["label_mask"].cpu().numpy()
                
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                
                all_labels.append(labels)
                all_preds.append(probs)
                all_masks.append(label_mask)
        
        # Concatenate
        import numpy as np
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        # Compute per-class AUC
        num_classes = all_labels.shape[1]
        class_aucs = []
        
        for i in range(num_classes):
            # Only compute AUC if we have both positive and negative samples
            mask = all_masks[:, i] == 1
            if mask.sum() == 0:
                continue
                
            y_true = all_labels[mask, i]
            y_pred = all_preds[mask, i]
            
            # Check if we have both classes
            if len(np.unique(y_true)) < 2:
                continue
            
            try:
                auc = roc_auc_score(y_true, y_pred)
                class_aucs.append(auc)
            except ValueError:
                continue
        
        # Macro average
        macro_auc = float(np.mean(class_aucs)) if class_aucs else 0.0
        
        return macro_auc

# Made with Bob
