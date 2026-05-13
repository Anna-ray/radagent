"""
radagent.modality.dicom_io
--------------------------
DICOM I/O utilities for universal modality routing.

Author: Rayane Aggoune
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import FileDataset


@dataclass
class DICOMMetadata:
    """DICOM metadata extracted from file."""
    modality: str
    body_part: str | None
    study_uid: str
    series_uid: str
    sop_uid: str
    patient_id: str | None
    study_description: str | None
    series_description: str | None
    manufacturer: str | None
    rows: int
    columns: int
    num_frames: int
    pixel_spacing: tuple[float, float] | None
    slice_thickness: float | None


def load_dicom(path: str | Path) -> dict[str, Any]:
    """Load DICOM file and extract metadata + pixel array.
    
    Args:
        path: Path to DICOM file
        
    Returns:
        Dictionary with:
        - pixel_array: numpy array (2D or 3D for multi-frame)
        - metadata: DICOMMetadata object
        - header_summary: dict of key DICOM tags
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")
    
    # Load DICOM
    try:
        ds: FileDataset = pydicom.dcmread(str(path))
    except Exception as e:
        raise ValueError(f"Failed to read DICOM file: {e}")
    
    # Extract pixel array
    try:
        pixel_array = ds.pixel_array
        
        # Handle multi-frame (CT, US, etc.)
        if len(pixel_array.shape) == 3:
            # (frames, rows, cols)
            num_frames = pixel_array.shape[0]
        else:
            # (rows, cols)
            num_frames = 1
            
    except Exception as e:
        raise ValueError(f"Failed to extract pixel array: {e}")
    
    # Extract metadata
    modality = str(ds.get("Modality", "OT"))
    body_part = str(ds.get("BodyPartExamined", "")).upper() if "BodyPartExamined" in ds else None
    
    # UIDs
    study_uid = str(ds.get("StudyInstanceUID", "unknown"))
    series_uid = str(ds.get("SeriesInstanceUID", "unknown"))
    sop_uid = str(ds.get("SOPInstanceUID", "unknown"))
    
    # Patient info (anonymize in production)
    patient_id = str(ds.get("PatientID", None)) if "PatientID" in ds else None
    
    # Descriptions
    study_description = str(ds.get("StudyDescription", None)) if "StudyDescription" in ds else None
    series_description = str(ds.get("SeriesDescription", None)) if "SeriesDescription" in ds else None
    
    # Equipment
    manufacturer = str(ds.get("Manufacturer", None)) if "Manufacturer" in ds else None
    
    # Image dimensions
    rows = int(ds.get("Rows", 0))
    columns = int(ds.get("Columns", 0))
    
    # Spacing
    pixel_spacing = None
    if "PixelSpacing" in ds:
        try:
            pixel_spacing = tuple(float(x) for x in ds.PixelSpacing)
        except:
            pass
    
    slice_thickness = None
    if "SliceThickness" in ds:
        try:
            slice_thickness = float(ds.SliceThickness)
        except:
            pass
    
    metadata = DICOMMetadata(
        modality=modality,
        body_part=body_part,
        study_uid=study_uid,
        series_uid=series_uid,
        sop_uid=sop_uid,
        patient_id=patient_id,
        study_description=study_description,
        series_description=series_description,
        manufacturer=manufacturer,
        rows=rows,
        columns=columns,
        num_frames=num_frames,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
    )
    
    # Header summary (for audit/logging)
    header_summary = {
        "modality": modality,
        "body_part": body_part,
        "study_uid": study_uid,
        "series_uid": series_uid,
        "rows": rows,
        "columns": columns,
        "num_frames": num_frames,
        "manufacturer": manufacturer,
    }
    
    return {
        "pixel_array": pixel_array,
        "metadata": metadata,
        "header_summary": header_summary,
    }


def is_dicom_file(path: str | Path) -> bool:
    """Check if file is a valid DICOM file.
    
    Args:
        path: Path to file
        
    Returns:
        True if valid DICOM
    """
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True)
        return True
    except:
        return False

# Made with Bob
