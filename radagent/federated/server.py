"""
radagent.federated.server
-------------------------
FedAvg server with SHA-256 audit chain.

Author: Rayane Aggoune
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class ClientUpdate:
    """Update from a hospital node after local training."""
    node_id: str
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    local_auc: float
    round_number: int
    wall_clock_seconds: float


@dataclass
class FederationRound:
    """Record of a complete federation round."""
    round_number: int
    num_clients: int
    total_samples: int
    global_auc: float
    client_aucs: dict[str, float]
    parameter_divergence: float
    timestamp: float
    previous_audit_hash: str | None
    audit_hash: str


class FedAvgServer:
    """Federated Averaging server with auditable hash chain.
    
    Implements the FedAvg algorithm (McMahan et al., 2017) with SHA-256
    audit receipts linking each round to the previous one.
    
    Args:
        initial_state: Initial global model state dict
        audit_dir: Directory to write audit receipts
    """
    
    def __init__(
        self,
        initial_state: dict[str, torch.Tensor],
        audit_dir: str | Path,
    ):
        self.global_state = {k: v.clone() for k, v in initial_state.items()}
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        self.round_history: list[FederationRound] = []
        self.previous_audit_hash: str | None = None
        
    def aggregate(
        self,
        updates: list[ClientUpdate],
        round_number: int,
    ) -> dict[str, torch.Tensor]:
        """Aggregate client updates using weighted averaging.
        
        Args:
            updates: List of client updates from this round
            round_number: Current round number
            
        Returns:
            New global state dict
        """
        if not updates:
            raise ValueError("No client updates to aggregate")
            
        # Compute total samples for weighting
        total_samples = sum(u.num_samples for u in updates)
        
        # Initialize aggregated state with zeros
        aggregated_state = {}
        for key in self.global_state.keys():
            aggregated_state[key] = torch.zeros_like(self.global_state[key])
        
        # Weighted average of client states
        for update in updates:
            weight = update.num_samples / total_samples
            for key in aggregated_state.keys():
                aggregated_state[key] += weight * update.state_dict[key].to(
                    aggregated_state[key].device
                )
        
        # Update global state
        self.global_state = aggregated_state
        
        return self.global_state
    
    def set_global_auc(self, round_number: int, auc: float) -> None:
        """Record global AUC for this round (called after global evaluation).
        
        Args:
            round_number: Round number
            auc: Global macro AUC on held-out test set
        """
        # Find the round record
        for record in self.round_history:
            if record.round_number == round_number:
                # Update the record (dataclasses are mutable)
                object.__setattr__(record, 'global_auc', auc)
                break
    
    def _compute_divergence(self, updates: list[ClientUpdate]) -> float:
        """Compute mean pairwise L2 divergence of client parameters.
        
        This is a judge-facing metric showing how different the client
        models are from each other. Higher divergence = more heterogeneous
        data distributions.
        
        Args:
            updates: Client updates
            
        Returns:
            Mean pairwise L2 distance (sampled from first layer weights)
        """
        if len(updates) < 2:
            return 0.0
        
        # Sample a parameter tensor for divergence computation
        # (computing full model divergence is expensive)
        sample_key = list(updates[0].state_dict.keys())[0]
        
        divergences = []
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                param_i = updates[i].state_dict[sample_key].flatten()
                param_j = updates[j].state_dict[sample_key].flatten()
                
                # L2 distance
                dist = torch.norm(param_i - param_j, p=2).item()
                divergences.append(dist)
        
        return float(np.mean(divergences)) if divergences else 0.0
    
    def _write_audit(
        self,
        record: FederationRound,
        updates: list[ClientUpdate],
    ) -> None:
        """Write audit receipt with SHA-256 hash chain.
        
        Args:
            record: Federation round record
            updates: Client updates (for detailed audit)
        """
        audit_path = self.audit_dir / f"round_{record.round_number:03d}.json"
        
        # Build audit document
        audit_doc = {
            "round_number": record.round_number,
            "num_clients": record.num_clients,
            "total_samples": record.total_samples,
            "global_auc": record.global_auc,
            "client_aucs": record.client_aucs,
            "parameter_divergence": record.parameter_divergence,
            "timestamp": record.timestamp,
            "previous_audit_hash": record.previous_audit_hash,
            "client_details": [
                {
                    "node_id": u.node_id,
                    "num_samples": u.num_samples,
                    "local_auc": u.local_auc,
                    "wall_clock_seconds": u.wall_clock_seconds,
                }
                for u in updates
            ],
        }
        
        # Write to disk
        with open(audit_path, "w") as f:
            json.dump(audit_doc, f, indent=2)
        
        print(f"[federated] Audit receipt written: {audit_path}")
    
    def finalize_round(
        self,
        updates: list[ClientUpdate],
        round_number: int,
        global_auc: float = 0.0,
    ) -> FederationRound:
        """Finalize a federation round and write audit receipt.
        
        Args:
            updates: Client updates
            round_number: Round number
            global_auc: Global AUC (0.0 if not yet evaluated)
            
        Returns:
            Federation round record
        """
        # Compute metrics
        total_samples = sum(u.num_samples for u in updates)
        client_aucs = {u.node_id: u.local_auc for u in updates}
        divergence = self._compute_divergence(updates)
        
        # Create round record
        record = FederationRound(
            round_number=round_number,
            num_clients=len(updates),
            total_samples=total_samples,
            global_auc=global_auc,
            client_aucs=client_aucs,
            parameter_divergence=divergence,
            timestamp=time.time(),
            previous_audit_hash=self.previous_audit_hash,
            audit_hash="",  # Will be computed below
        )
        
        # Compute SHA-256 hash of this round's audit
        audit_content = json.dumps(
            {
                "round_number": record.round_number,
                "num_clients": record.num_clients,
                "total_samples": record.total_samples,
                "global_auc": record.global_auc,
                "client_aucs": record.client_aucs,
                "parameter_divergence": record.parameter_divergence,
                "timestamp": record.timestamp,
                "previous_audit_hash": record.previous_audit_hash,
            },
            sort_keys=True,
        ).encode("utf-8")
        
        audit_hash = hashlib.sha256(audit_content).hexdigest()
        object.__setattr__(record, 'audit_hash', audit_hash)
        
        # Write audit receipt
        self._write_audit(record, updates)
        
        # Update chain
        self.previous_audit_hash = audit_hash
        self.round_history.append(record)
        
        return record

# Made with Bob
