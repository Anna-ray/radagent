"""
tests/test_federated.py
-----------------------
Tests for federated learning components.

Author: Rayane Aggoune
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from radagent.federated import ClientUpdate, FedAvgServer


class TinyModel(nn.Module):
    """Tiny model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def test_fedavg_weighted_average_correctness():
    """Test that FedAvg correctly computes weighted average."""
    # Create initial model
    model = TinyModel()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Create two client updates with known weights
    state_a = {k: torch.ones_like(v) for k, v in initial_state.items()}
    state_b = {k: torch.ones_like(v) * 2.0 for k, v in initial_state.items()}
    
    update_a = ClientUpdate(
        node_id="A",
        state_dict=state_a,
        num_samples=100,
        local_auc=0.8,
        round_number=1,
        wall_clock_seconds=10.0,
    )
    
    update_b = ClientUpdate(
        node_id="B",
        state_dict=state_b,
        num_samples=200,
        local_auc=0.85,
        round_number=1,
        wall_clock_seconds=15.0,
    )
    
    # Create server
    with tempfile.TemporaryDirectory() as tmpdir:
        server = FedAvgServer(initial_state, tmpdir)
        
        # Aggregate
        aggregated = server.aggregate([update_a, update_b], round_number=1)
        
        # Expected: (100*1 + 200*2) / 300 = 500/300 = 1.6667
        for key in aggregated.keys():
            expected = torch.ones_like(aggregated[key]) * (500.0 / 300.0)
            assert torch.allclose(aggregated[key], expected, atol=1e-4)


def test_audit_receipt_schema_and_hash_chain():
    """Test that audit receipts have correct schema and hash chain."""
    model = TinyModel()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        server = FedAvgServer(initial_state, tmpdir)
        
        # Create dummy updates
        update = ClientUpdate(
            node_id="A",
            state_dict={k: v.clone() for k, v in initial_state.items()},
            num_samples=100,
            local_auc=0.8,
            round_number=1,
            wall_clock_seconds=10.0,
        )
        
        # Round 1
        server.aggregate([update], round_number=1)
        record1 = server.finalize_round([update], round_number=1, global_auc=0.75)
        
        # Check audit file exists
        audit_path = Path(tmpdir) / "round_001.json"
        assert audit_path.exists()
        
        # Check schema
        with open(audit_path) as f:
            audit = json.load(f)
        
        assert "round_number" in audit
        assert "num_clients" in audit
        assert "total_samples" in audit
        assert "global_auc" in audit
        assert "client_aucs" in audit
        assert "parameter_divergence" in audit
        assert "timestamp" in audit
        assert "previous_audit_hash" in audit
        assert "client_details" in audit
        
        # First round should have no previous hash
        assert audit["previous_audit_hash"] is None
        
        # Round 2
        update.round_number = 2
        server.aggregate([update], round_number=2)
        record2 = server.finalize_round([update], round_number=2, global_auc=0.78)
        
        # Check hash chain
        assert record2.previous_audit_hash == record1.audit_hash
        
        # Check audit file
        audit_path2 = Path(tmpdir) / "round_002.json"
        with open(audit_path2) as f:
            audit2 = json.load(f)
        
        assert audit2["previous_audit_hash"] == record1.audit_hash


def test_label_harmonization_chexpert():
    """Test that CheXpert labels are correctly harmonized to NIH-14 schema."""
    from radagent.data.cxr_datasets import CLASS_NAMES
    
    # Verify class order is immutable
    expected_order = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]
    
    assert CLASS_NAMES == expected_order
    
    # Test mapping
    assert "Effusion" in CLASS_NAMES  # CheXpert "Pleural Effusion" -> "Effusion"
    assert "Infiltration" in CLASS_NAMES  # CheXpert "Lung Opacity" -> "Infiltration"


def test_no_raw_data_in_audit_log():
    """Test that audit logs contain no raw patient data."""
    model = TinyModel()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        server = FedAvgServer(initial_state, tmpdir)
        
        update = ClientUpdate(
            node_id="Hospital_A",
            state_dict={k: v.clone() for k, v in initial_state.items()},
            num_samples=5000,
            local_auc=0.82,
            round_number=1,
            wall_clock_seconds=120.0,
        )
        
        server.aggregate([update], round_number=1)
        server.finalize_round([update], round_number=1, global_auc=0.80)
        
        # Read audit file
        audit_path = Path(tmpdir) / "round_001.json"
        with open(audit_path) as f:
            audit_content = f.read()
        
        # Verify no image data, no pixel values, no patient IDs
        forbidden_terms = [
            "pixel",
            "image_data",
            "patient_id",
            "dicom",
            "array",
            "tensor",
        ]
        
        for term in forbidden_terms:
            assert term.lower() not in audit_content.lower()
        
        # Verify only metadata is present
        audit = json.loads(audit_content)
        assert "num_samples" in audit["client_details"][0]
        assert "local_auc" in audit["client_details"][0]
        assert "wall_clock_seconds" in audit["client_details"][0]


def test_parameter_divergence_computation():
    """Test that parameter divergence is computed correctly."""
    model = TinyModel()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Create two updates with different parameters
    state_a = {k: v.clone() for k, v in initial_state.items()}
    state_b = {k: v.clone() + 1.0 for k, v in initial_state.items()}
    
    update_a = ClientUpdate(
        node_id="A",
        state_dict=state_a,
        num_samples=100,
        local_auc=0.8,
        round_number=1,
        wall_clock_seconds=10.0,
    )
    
    update_b = ClientUpdate(
        node_id="B",
        state_dict=state_b,
        num_samples=100,
        local_auc=0.8,
        round_number=1,
        wall_clock_seconds=10.0,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        server = FedAvgServer(initial_state, tmpdir)
        server.aggregate([update_a, update_b], round_number=1)
        record = server.finalize_round([update_a, update_b], round_number=1, global_auc=0.8)
        
        # Divergence should be > 0 since parameters differ
        assert record.parameter_divergence > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
