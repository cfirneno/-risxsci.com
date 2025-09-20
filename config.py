    #!/usr/bin/env python3
"""
Configuration Module
===================

Configuration classes, enums, and constants for the cancer analysis platform.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

class CancerType(Enum):
    """Cancer type enumeration"""
    BREAST = "breast"
    LUNG = "lung"
    COLON = "colon"
    PROSTATE = "prostate"
    PANCREATIC = "pancreatic"
    MELANOMA = "melanoma"
    OVARIAN = "ovarian"
    BLADDER = "bladder"
    KIDNEY = "kidney"
    LIVER = "liver"

class AnalysisType(Enum):
    """Analysis type enumeration"""
    WSI = "wsi"
    NGS = "ngs"
    METHYLATION = "methylation"
    MRD = "mrd"
    RISK = "risk"
    CLINICAL = "clinical"

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AnalysisConfig:
    """Base analysis configuration"""
    analysis_type: AnalysisType
    cancer_type: Optional[CancerType] = None
    patient_id: Optional[str] = None
    timestamp: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

@dataclass
class WSIConfig:
    """WSI Analysis configuration"""
    patch_size: int = 512
    overlap: float = 0.1
    magnification: str = "20x"
    enhancement: bool = True
    model_type: str = "flow_matching"

@dataclass
class NGSConfig:
    """NGS Analysis configuration"""
    variant_filter: str = "high_quality"
    annotation_source: str = "clinvar"
    min_coverage: int = 20
    min_vaf: float = 0.05
    reference_genome: str = "GRCh38"

@dataclass
class MRDConfig:
    """MRD Detection configuration"""
    sample_type: str = "plasma"
    detection_threshold: float = 0.01
    tracking_mutations: List[str] = None
    panel_type: str = "comprehensive"

@dataclass
class RiskConfig:
    """Risk Stratification configuration"""
    model_version: str = "v2.1"
    include_genomics: bool = True
    include_imaging: bool = True
    time_horizon: str = "5_year"

# Platform constants
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.svs']
SUPPORTED_VCF_FORMATS = ['.vcf', '.vcf.gz']
SUPPORTED_METHYLATION_FORMATS = ['.csv', '.txt', '.idat']

DEFAULT_MUTATIONS = [
    'EGFR', 'KRAS', 'TP53', 'PIK3CA', 'BRAF', 'APC', 'PTEN', 'MYC',
    'RB1', 'BRCA1', 'BRCA2', 'MLH1', 'MSH2', 'MSH6', 'PMS2'
]

# Analysis thresholds
RISK_THRESHOLDS = {
    RiskLevel.LOW: 0.3,
    RiskLevel.MODERATE: 0.7,
    RiskLevel.HIGH: 0.9,
    RiskLevel.CRITICAL: 1.0
}

# Model parameters
FLOW_MATCHING_PARAMS = {
    'num_steps': 1000,
    'sigma_min': 0.002,
    'sigma_max': 80.0,
    'rho': 7.0,
    'solver': 'heun'
}

LOGISTIC_DYNAMICS_PARAMS = {
    'growth_rate': 0.1,
    'carrying_capacity': 1.0,
    'initial_value': 0.01,
    'noise_scale': 0.05
}
