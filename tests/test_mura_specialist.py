"""
Tests for MURA Bone X-ray Specialist

Author: Rayane Aggoune
"""

import pytest
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

from radagent.specialists.mura import predict, MURAInferenceResult
from radagent.specialists.mura.dataset import MURADataset, get_mura_statistics


class TestMURAInference:
    """Test MURA specialist inference interface."""
    
    def test_predict_returns_expected_schema(self, tmp_path):
        """Test that predict() returns the correct schema."""
        # Create a synthetic bone X-ray image
        img_path = tmp_path / "wrist_test.png"
        img = Image.new("RGB", (320, 320), color=(128, 128, 128))
        img.save(img_path)
        
        # Run inference
        result = predict(str(img_path))
        
        # Verify schema
        assert isinstance(result, MURAInferenceResult)
        assert result.body_part in ["SHOULDER", "HUMERUS", "ELBOW", "FOREARM", 
                                     "WRIST", "HAND", "FINGER"]
        assert 0.0 <= result.abnormality_probability <= 1.0
        assert 0.0 <= result.threshold <= 1.0
        assert isinstance(result.above_threshold, bool)
        assert result.calibration_band in ["high", "medium", "low"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.status == "REGISTERED (placeholder - specialist training scheduled v2.1)"
        assert result.model_version == "mura_v1_placeholder"
        assert result.processing_time_ms > 0
    
    def test_predict_with_dicom_metadata(self, tmp_path):
        """Test body part detection from DICOM metadata."""
        img_path = tmp_path / "elbow_test.png"
        img = Image.new("RGB", (320, 320), color=(100, 100, 100))
        img.save(img_path)
        
        # Provide DICOM metadata
        dicom_metadata = {
            "BodyPartExamined": "ELBOW",
            "Modality": "DX"
        }
        
        result = predict(str(img_path), dicom_metadata=dicom_metadata)
        
        assert result.body_part == "ELBOW"
    
    def test_predict_with_gradcam_request(self, tmp_path):
        """Test Grad-CAM path generation."""
        img_path = tmp_path / "shoulder_test.png"
        img = Image.new("RGB", (320, 320), color=(150, 150, 150))
        img.save(img_path)
        
        output_dir = tmp_path / "gradcam_output"
        
        result = predict(
            str(img_path),
            generate_gradcam=True,
            output_dir=str(output_dir)
        )
        
        # Grad-CAM path should be set (even if placeholder)
        assert result.gradcam_path is not None
        assert "gradcam" in result.gradcam_path.lower()
        assert "placeholder" in result.gradcam_path.lower()
    
    def test_placeholder_returns_conservative_predictions(self, tmp_path):
        """Test that placeholder returns conservative (low confidence) predictions."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (320, 320), color=(128, 128, 128))
        img.save(img_path)
        
        result = predict(str(img_path))
        
        # Placeholder should be conservative
        assert result.confidence < 0.60  # Low confidence
        assert result.calibration_band == "low"  # Honest uncertainty
        assert "placeholder" in result.status.lower()
    
    def test_result_serialization(self, tmp_path):
        """Test that results can be serialized to JSON."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (320, 320), color=(128, 128, 128))
        img.save(img_path)
        
        result = predict(str(img_path))
        
        # Test dict conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "body_part" in result_dict
        assert "abnormality_probability" in result_dict
        
        # Test JSON conversion
        result_json = result.to_json()
        assert isinstance(result_json, str)
        assert "body_part" in result_json


class TestMURADataset:
    """Test MURA dataset loader."""
    
    def test_dataset_initialization_without_data(self, tmp_path):
        """Test that dataset initializes gracefully when MURA data is not available."""
        # Create empty directory structure
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        
        dataset = MURADataset(str(tmp_path), split="train")
        
        # Should initialize with zero studies
        assert len(dataset) == 0
    
    def test_dataset_with_synthetic_structure(self, tmp_path):
        """Test dataset loading with synthetic MURA directory structure."""
        # Create synthetic MURA structure
        train_dir = tmp_path / "train"
        body_part_dir = train_dir / "XR_WRIST"
        patient_dir = body_part_dir / "patient00001"
        study_dir = patient_dir / "study1_positive"
        study_dir.mkdir(parents=True)
        
        # Create synthetic images
        for i in range(3):
            img = Image.new("RGB", (320, 320), color=(100 + i * 10, 100, 100))
            img.save(study_dir / f"image{i}.png")
        
        dataset = MURADataset(str(tmp_path), split="train")
        
        # Should load one study
        assert len(dataset) == 1
        
        # Get the study
        study = dataset[0]
        assert study["body_part"] == "WRIST"
        assert study["label"] == 1  # positive
        assert len(study["images"]) == 3
    
    def test_statistics_with_empty_dataset(self, tmp_path):
        """Test statistics computation with empty dataset."""
        train_dir = tmp_path / "train"
        valid_dir = tmp_path / "valid"
        train_dir.mkdir()
        valid_dir.mkdir()
        
        stats = get_mura_statistics(str(tmp_path))
        
        assert stats["num_train_studies"] == 0
        assert stats["num_valid_studies"] == 0
        assert stats["class_distribution"]["normal"] == 0
        assert stats["class_distribution"]["abnormal"] == 0


class TestMURAIntegration:
    """Integration tests for MURA specialist."""
    
    def test_end_to_end_inference_pipeline(self, tmp_path):
        """Test complete inference pipeline from image to result."""
        # Create test image
        img_path = tmp_path / "bone_xray.png"
        img = Image.new("RGB", (320, 320), color=(128, 128, 128))
        img.save(img_path)
        
        # Run inference with all options
        result = predict(
            str(img_path),
            dicom_metadata={"BodyPartExamined": "HAND"},
            generate_gradcam=True,
            output_dir=str(tmp_path / "output")
        )
        
        # Verify complete result
        assert result.body_part == "HAND"
        assert result.status.startswith("REGISTERED")
        assert result.gradcam_path is not None
        
        # Verify JSON serialization works
        json_str = result.to_json()
        assert "HAND" in json_str
        assert "placeholder" in json_str.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
