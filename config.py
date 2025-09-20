"""
Configuration Module - Cancer Analysis Platform
==============================================

This module contains all configuration settings, enums, and constants
for the sophisticated cancer analysis platform.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
import multiprocessing as mp

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Constants
CANCER_TYPES_COUNT = 47
MUTATION_FEATURES = 2048
EXPRESSION_FEATURES = 20531
PATHOLOGY_FEATURES = 512
CLINICAL_FEATURES = 156
NUM_WORKERS = mp.cpu_count()
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Directory paths
MODEL_CHECKPOINT_DIR = "models"
DATA_CACHE_DIR = "cache"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

class CancerType(Enum):
    """Comprehensive cancer type enumeration with ICD-10 codes and detailed classification"""
    
    # Lung cancers (C78.x)
    LUNG_ADENOCARCINOMA = ("C78.00", "Lung adenocarcinoma", "lung", "adenocarcinoma", 0.45, ["EGFR", "KRAS", "ALK"], 5)
    LUNG_SQUAMOUS = ("C78.01", "Lung squamous cell carcinoma", "lung", "squamous", 0.30, ["PIK3CA", "FGFR1", "SOX2"], 4)
    LUNG_SMALL_CELL = ("C78.02", "Small cell lung carcinoma", "lung", "neuroendocrine", 0.15, ["RB1", "TP53", "MYC"], 3)
    LUNG_LARGE_CELL = ("C78.03", "Large cell lung carcinoma", "lung", "large_cell", 0.10, ["STK11", "KEAP1", "NF1"], 4)
    LUNG_CARCINOID = ("C78.04", "Lung carcinoid", "lung", "carcinoid", 0.02, ["MEN1", "DAXX", "ATRX"], 7)
    
    # Breast cancers (C50.x)
    BREAST_DUCTAL = ("C50.911", "Invasive ductal carcinoma", "breast", "ductal", 0.75, ["ESR1", "ERBB2", "PIK3CA"], 6)
    BREAST_LOBULAR = ("C50.912", "Invasive lobular carcinoma", "breast", "lobular", 0.15, ["CDH1", "PIK3CA", "ESR1"], 6)
    BREAST_INFLAMMATORY = ("C50.913", "Inflammatory breast carcinoma", "breast", "inflammatory", 0.02, ["ERBB2", "TP53", "BRCA1"], 2)
    BREAST_TRIPLE_NEGATIVE = ("C50.914", "Triple negative breast carcinoma", "breast", "basal", 0.15, ["TP53", "BRCA1", "PIK3CA"], 3)
    BREAST_HER2_POSITIVE = ("C50.915", "HER2 positive breast carcinoma", "breast", "her2", 0.20, ["ERBB2", "PIK3CA", "TP53"], 4)
    
    # Colorectal cancers (C18.x, C19.x, C20.x)
    COLORECTAL_ADENOCARCINOMA = ("C18.9", "Colorectal adenocarcinoma", "colorectal", "adenocarcinoma", 0.85, ["APC", "TP53", "KRAS"], 5)
    COLORECTAL_MUCINOUS = ("C18.8", "Mucinous colorectal carcinoma", "colorectal", "mucinous", 0.10, ["BRAF", "MLH1", "MSH2"], 4)
    COLORECTAL_SIGNET_RING = ("C18.7", "Signet ring colorectal carcinoma", "colorectal", "signet_ring", 0.02, ["CDH1", "TP53", "PIK3CA"], 3)
    
    # Prostate cancers (C61.x)
    PROSTATE_ADENOCARCINOMA = ("C61.9", "Prostate adenocarcinoma", "prostate", "adenocarcinoma", 0.95, ["AR", "PTEN", "TP53"], 6)
    PROSTATE_NEUROENDOCRINE = ("C61.8", "Prostate neuroendocrine", "prostate", "neuroendocrine", 0.02, ["RB1", "TP53", "MYCN"], 2)
    PROSTATE_DUCTAL = ("C61.7", "Prostate ductal adenocarcinoma", "prostate", "ductal", 0.02, ["AR", "PTEN", "SPOP"], 4)
    
    # Liver cancers (C22.x)
    LIVER_HEPATOCELLULAR = ("C22.0", "Hepatocellular carcinoma", "liver", "hepatocellular", 0.80, ["TP53", "CTNNB1", "AXIN1"], 4)
    LIVER_CHOLANGIOCARCINOMA = ("C22.1", "Cholangiocarcinoma", "liver", "cholangiocarcinoma", 0.15, ["IDH1", "FGFR2", "BRAF"], 3)
    LIVER_ANGIOSARCOMA = ("C22.3", "Liver angiosarcoma", "liver", "angiosarcoma", 0.02, ["TP53", "RB1", "CDKN2A"], 2)
    
    # Pancreatic cancers (C25.x)
    PANCREATIC_DUCTAL = ("C25.9", "Pancreatic ductal adenocarcinoma", "pancreas", "ductal", 0.85, ["KRAS", "TP53", "CDKN2A"], 2)
    PANCREATIC_NEUROENDOCRINE = ("C25.8", "Pancreatic neuroendocrine", "pancreas", "neuroendocrine", 0.05, ["MEN1", "DAXX", "ATRX"], 5)
    PANCREATIC_ACINAR = ("C25.7", "Pancreatic acinar carcinoma", "pancreas", "acinar", 0.02, ["APC", "TP53", "SMAD4"], 3)
    
    # Brain cancers (C71.x)
    BRAIN_GLIOBLASTOMA = ("C71.9", "Glioblastoma multiforme", "brain", "glioblastoma", 0.50, ["IDH1", "TP53", "EGFR"], 1)
    BRAIN_OLIGODENDROGLIOMA = ("C71.8", "Oligodendroglioma", "brain", "oligodendroglioma", 0.05, ["IDH1", "1p19q", "CIC"], 7)
    BRAIN_MENINGIOMA = ("C70.9", "Meningioma", "brain", "meningioma", 0.35, ["NF2", "SMO", "AKT1"], 8)
    BRAIN_MEDULLOEPITHELIOMA = ("C71.7", "Medulloepithelioma", "brain", "medulloepithelioma", 0.01, ["TP53", "RB1", "MYC"], 2)
    
    # Kidney cancers (C64.x)
    KIDNEY_CLEAR_CELL = ("C64.9", "Clear cell renal carcinoma", "kidney", "clear_cell", 0.70, ["VHL", "PBRM1", "SETD2"], 5)
    KIDNEY_PAPILLARY = ("C64.8", "Papillary renal carcinoma", "kidney", "papillary", 0.15, ["MET", "CDKN2A", "NF2"], 5)
    KIDNEY_CHROMOPHOBE = ("C64.7", "Chromophobe renal carcinoma", "kidney", "chromophobe", 0.05, ["TP53", "PTEN", "CDKN2A"], 6)
    
    # Bladder cancers (C67.x)
    BLADDER_UROTHELIAL = ("C67.9", "Urothelial bladder carcinoma", "bladder", "urothelial", 0.90, ["TP53", "RB1", "FGFR3"], 4)
    BLADDER_SQUAMOUS = ("C67.8", "Squamous cell bladder carcinoma", "bladder", "squamous", 0.05, ["TP53", "CDKN2A", "PIK3CA"], 3)
    BLADDER_ADENOCARCINOMA = ("C67.7", "Bladder adenocarcinoma", "bladder", "adenocarcinoma", 0.02, ["TP53", "KRAS", "ERBB2"], 3)
    
    # Ovarian cancers (C56.x)
    OVARIAN_SEROUS = ("C56.9", "High-grade serous ovarian carcinoma", "ovary", "serous", 0.70, ["TP53", "BRCA1", "BRCA2"], 3)
    OVARIAN_ENDOMETRIOID = ("C56.8", "Endometrioid ovarian carcinoma", "ovary", "endometrioid", 0.10, ["CTNNB1", "PIK3CA", "PTEN"], 5)
    OVARIAN_CLEAR_CELL = ("C56.7", "Clear cell ovarian carcinoma", "ovary", "clear_cell", 0.05, ["PIK3CA", "ARID1A", "PPP2R1A"], 4)
    OVARIAN_MUCINOUS = ("C56.6", "Mucinous ovarian carcinoma", "ovary", "mucinous", 0.03, ["KRAS", "TP53", "CDKN2A"], 4)
    
    # Cervical cancers (C53.x)
    CERVICAL_SQUAMOUS = ("C53.9", "Cervical squamous cell carcinoma", "cervix", "squamous", 0.80, ["TP53", "PIK3CA", "PTEN"], 4)
    CERVICAL_ADENOCARCINOMA = ("C53.8", "Cervical adenocarcinoma", "cervix", "adenocarcinoma", 0.15, ["TP53", "PIK3CA", "KRAS"], 4)
    CERVICAL_ADENOSQUAMOUS = ("C53.7", "Cervical adenosquamous carcinoma", "cervix", "adenosquamous", 0.03, ["TP53", "PIK3CA", "FBXW7"], 3)
    
    # Stomach cancers (C16.x)
    STOMACH_ADENOCARCINOMA = ("C16.9", "Gastric adenocarcinoma", "stomach", "adenocarcinoma", 0.90, ["TP53", "PIK3CA", "ARID1A"], 4)
    STOMACH_SIGNET_RING = ("C16.8", "Gastric signet ring carcinoma", "stomach", "signet_ring", 0.07, ["CDH1", "TP53", "PIK3CA"], 3)
    STOMACH_LYMPHOMA = ("C16.7", "Gastric lymphoma", "stomach", "lymphoma", 0.02, ["MYC", "BCL2", "TP53"], 5)
    
    # Esophageal cancers (C15.x)
    ESOPHAGEAL_SQUAMOUS = ("C15.9", "Esophageal squamous cell carcinoma", "esophagus", "squamous", 0.50, ["TP53", "CDKN2A", "PIK3CA"], 3)
    ESOPHAGEAL_ADENOCARCINOMA = ("C15.8", "Esophageal adenocarcinoma", "esophagus", "adenocarcinoma", 0.45, ["TP53", "CDKN2A", "SMAD4"], 3)
    
    # Blood cancers (C81.x-C96.x)
    LEUKEMIA_AML = ("C92.0", "Acute myeloid leukemia", "blood", "myeloid", 0.25, ["FLT3", "NPM1", "CEBPA"], 3)
    LEUKEMIA_ALL = ("C91.0", "Acute lymphoid leukemia", "blood", "lymphoid", 0.20, ["BCR-ABL1", "ETV6", "PAX5"], 5)
    LEUKEMIA_CML = ("C92.1", "Chronic myeloid leukemia", "blood", "myeloid_chronic", 0.15, ["BCR-ABL1", "TP53", "RUNX1"], 6)
    LEUKEMIA_CLL = ("C91.1", "Chronic lymphoid leukemia", "blood", "lymphoid_chronic", 0.25, ["TP53", "ATM", "NOTCH1"], 7)
    LYMPHOMA_HODGKIN = ("C81.9", "Hodgkin lymphoma", "lymph", "hodgkin", 0.10, ["REL", "TNFAIP3", "B2M"], 8)
    LYMPHOMA_NHL = ("C85.9", "Non-Hodgkin lymphoma", "lymph", "non_hodgkin", 0.85, ["MYC", "BCL2", "BCL6"], 6)
    
    def __init__(self, icd_code, full_name, organ, histology, prevalence, key_mutations, survival_years):
        self.icd_code = icd_code
        self.full_name = full_name
        self.organ = organ
        self.histology = histology
        self.prevalence = prevalence
        self.key_mutations = key_mutations
        self.survival_years = survival_years

class TumorStage(Enum):
    """TNM staging system for cancer"""
    STAGE_0 = ("Tis N0 M0", "Carcinoma in situ", 0.98, 10)
    STAGE_IA = ("T1a N0 M0", "Early stage, small tumor", 0.95, 9)
    STAGE_IB = ("T1b N0 M0", "Early stage, slightly larger", 0.92, 8)
    STAGE_IIA = ("T2a N0 M0", "Moderate size, no spread", 0.85, 7)
    STAGE_IIB = ("T2b N1 M0", "Moderate size, lymph node", 0.75, 6)
    STAGE_IIIA = ("T3a N1 M0", "Large tumor, lymph nodes", 0.60, 5)
    STAGE_IIIB = ("T3b N2 M0", "Large tumor, multiple nodes", 0.45, 4)
    STAGE_IIIC = ("T4 N2 M0", "Very large, extensive nodes", 0.30, 3)
    STAGE_IV = ("T4 N3 M1", "Metastatic disease", 0.15, 2)
    
    def __init__(self, tnm, description, survival_rate, median_survival_months):
        self.tnm = tnm
        self.description = description
        self.survival_rate = survival_rate
        self.median_survival_months = median_survival_months

class GeneticMarker(Enum):
    """Key genetic markers and biomarkers for cancer analysis"""
    
    # Tumor suppressor genes
    TP53 = ("TP53", "tumor_suppressor", "Guardian of genome", 0.50, "high", ["DNA repair", "Apoptosis"])
    RB1 = ("RB1", "tumor_suppressor", "Cell cycle control", 0.15, "high", ["Cell cycle"])
    APC = ("APC", "tumor_suppressor", "Wnt pathway regulation", 0.80, "high", ["Cell adhesion"])
    PTEN = ("PTEN", "tumor_suppressor", "PI3K pathway inhibitor", 0.30, "high", ["Cell growth"])
    BRCA1 = ("BRCA1", "tumor_suppressor", "DNA repair", 0.70, "high", ["Homologous recombination"])
    BRCA2 = ("BRCA2", "tumor_suppressor", "DNA repair", 0.65, "high", ["Homologous recombination"])
    VHL = ("VHL", "tumor_suppressor", "Oxygen sensing", 0.90, "high", ["Angiogenesis"])
    NF1 = ("NF1", "tumor_suppressor", "RAS regulation", 0.25, "medium", ["Cell proliferation"])
    NF2 = ("NF2", "tumor_suppressor", "Cell contact inhibition", 0.60, "medium", ["Cell adhesion"])
    CDKN2A = ("CDKN2A", "tumor_suppressor", "Cell cycle inhibitor", 0.40, "high", ["Cell cycle"])
    
    # Oncogenes
    KRAS = ("KRAS", "oncogene", "Cell proliferation", 0.30, "high", ["MAPK pathway"])
    EGFR = ("EGFR", "oncogene", "Growth factor receptor", 0.25, "high", ["Cell proliferation"])
    MYC = ("MYC", "oncogene", "Transcription factor", 0.20, "high", ["Cell proliferation", "Apoptosis"])
    ERBB2 = ("ERBB2", "oncogene", "Growth factor receptor", 0.20, "high", ["Cell proliferation"])
    PIK3CA = ("PIK3CA", "oncogene", "PI3K pathway", 0.35, "high", ["Cell growth"])
    BRAF = ("BRAF", "oncogene", "MAPK pathway", 0.15, "high", ["Cell proliferation"])
    ALK = ("ALK", "oncogene", "Receptor tyrosine kinase", 0.05, "medium", ["Cell proliferation"])
    RET = ("RET", "oncogene", "Receptor tyrosine kinase", 0.02, "medium", ["Cell proliferation"])
    MET = ("MET", "oncogene", "Growth factor receptor", 0.10, "medium", ["Cell proliferation"])
    FGFR1 = ("FGFR1", "oncogene", "Growth factor receptor", 0.08, "medium", ["Cell proliferation"])
    
    # DNA repair genes
    MLH1 = ("MLH1", "dna_repair", "Mismatch repair", 0.15, "high", ["DNA repair"])
    MSH2 = ("MSH2", "dna_repair", "Mismatch repair", 0.12, "high", ["DNA repair"])
    MSH6 = ("MSH6", "dna_repair", "Mismatch repair", 0.10, "medium", ["DNA repair"])
    PMS2 = ("PMS2", "dna_repair", "Mismatch repair", 0.08, "medium", ["DNA repair"])
    ATM = ("ATM", "dna_repair", "DNA damage response", 0.20, "high", ["DNA repair"])
    CHEK2 = ("CHEK2", "dna_repair", "Cell cycle checkpoint", 0.15, "medium", ["DNA repair"])
    
    # Chromatin remodeling
    ARID1A = ("ARID1A", "chromatin", "SWI/SNF complex", 0.25, "medium", ["Chromatin remodeling"])
    SMARCA4 = ("SMARCA4", "chromatin", "SWI/SNF complex", 0.15, "medium", ["Chromatin remodeling"])
    KMT2D = ("KMT2D", "chromatin", "Histone methyltransferase", 0.20, "medium", ["Chromatin remodeling"])
    
    # Metabolism
    IDH1 = ("IDH1", "metabolism", "Isocitrate dehydrogenase", 0.75, "high", ["Metabolism"])
    IDH2 = ("IDH2", "metabolism", "Isocitrate dehydrogenase", 0.20, "medium", ["Metabolism"])
    
    def __init__(self, gene_name, gene_type, description, mutation_frequency, clinical_significance, pathways):
        self.gene_name = gene_name
        self.gene_type = gene_type
        self.description = description
        self.mutation_frequency = mutation_frequency
        self.clinical_significance = clinical_significance
        self.pathways = pathways

class TreatmentResponse(Enum):
    """Treatment response categories (RECIST criteria)"""
    COMPLETE_RESPONSE = ("CR", "Complete disappearance", 1.0, 24)
    PARTIAL_RESPONSE = ("PR", "≥30% decrease in target lesions", 0.8, 18)
    STABLE_DISEASE = ("SD", "<30% decrease or <20% increase", 0.6, 12)
    PROGRESSIVE_DISEASE = ("PD", "≥20% increase or new lesions", 0.2, 6)
    
    def __init__(self, code, description, benefit_score, median_months):
        self.code = code
        self.description = description
        self.benefit_score = benefit_score
        self.median_months = median_months

class DrugClass(Enum):
    """Major cancer drug classes with mechanisms"""
    CHEMOTHERAPY = ("chemo", "DNA damaging agents", 0.40, ["DNA damage"])
    TARGETED_THERAPY = ("targeted", "Specific molecular targets", 0.60, ["Pathway inhibition"])
    IMMUNOTHERAPY = ("immuno", "Immune system activation", 0.25, ["Immune activation"])
    HORMONE_THERAPY = ("hormone", "Hormone pathway blocking", 0.70, ["Hormone signaling"])
    RADIATION_THERAPY = ("radiation", "High-energy radiation", 0.45, ["DNA damage"])
    
    def __init__(self, category, description, response_rate, mechanisms):
        self.category = category
        self.description = description
        self.response_rate = response_rate
        self.mechanisms = mechanisms

@dataclass
class ModelConfig:
    """Configuration for the cancer analysis models"""
    
    # Model architecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    
    # Data parameters
    sequence_length: int = 1024
    num_features: int = 2048
    num_classes: int = len(CancerType)
    
    # Flow matching parameters
    flow_steps: int = 100
    beta_min: float = 0.1
    beta_max: float = 20.0
    
    # WSI parameters
    patch_size: int = 224
    num_patches: int = 36
    wsi_features: int = 2048
    
    # NGS parameters
    mutation_features: int = 2048
    expression_features: int = 20531
    
    # Clinical parameters
    clinical_features: int = 156
    
    # Risk analysis
    risk_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.3,
        'moderate': 0.7,
        'high': 1.0
    })
    
    # Economic parameters
    cost_effectiveness_threshold: float = 50000.0  # USD per QALY
    
    # Device settings
    device: str = str(DEVICE)
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 100
    save_interval: int = 1000
    
    # Validation
    validation_split: float = 0.2
    test_split: float = 0.1

# Biomarker thresholds and normal ranges
BIOMARKER_RANGES = {
    'PSA': {'normal': (0, 4), 'elevated': (4, 10), 'high': (10, float('inf')), 'units': 'ng/mL'},
    'CEA': {'normal': (0, 3), 'elevated': (3, 10), 'high': (10, float('inf')), 'units': 'ng/mL'},
    'CA125': {'normal': (0, 35), 'elevated': (35, 100), 'high': (100, float('inf')), 'units': 'U/mL'},
    'CA19-9': {'normal': (0, 37), 'elevated': (37, 100), 'high': (100, float('inf')), 'units': 'U/mL'},
    'AFP': {'normal': (0, 10), 'elevated': (10, 50), 'high': (50, float('inf')), 'units': 'ng/mL'},
    'HCG': {'normal': (0, 5), 'elevated': (5, 50), 'high': (50, float('inf')), 'units': 'mIU/mL'},
    'LDH': {'normal': (140, 280), 'elevated': (280, 500), 'high': (500, float('inf')), 'units': 'U/L'},
    'alkaline_phosphatase': {'normal': (44, 147), 'elevated': (147, 300), 'high': (300, float('inf')), 'units': 'U/L'}
}

# Treatment drug mappings
TREATMENT_MAPPINGS = {
    CancerType.LUNG_ADENOCARCINOMA: {
        'first_line': ['carboplatin', 'paclitaxel', 'bevacizumab'],
        'targeted': ['erlotinib', 'gefitinib', 'osimertinib'],
        'immunotherapy': ['pembrolizumab', 'nivolumab', 'atezolizumab']
    },
    CancerType.BREAST_DUCTAL: {
        'hormone_positive': ['tamoxifen', 'anastrozole', 'letrozole'],
        'her2_positive': ['trastuzumab', 'pertuzumab', 'T-DM1'],
        'triple_negative': ['carboplatin', 'doxorubicin', 'cyclophosphamide']
    },
    CancerType.COLORECTAL_ADENOCARCINOMA: {
        'first_line': ['5-fluorouracil', 'oxaliplatin', 'irinotecan'],
        'targeted': ['cetuximab', 'bevacizumab', 'panitumumab'],
        'immunotherapy': ['pembrolizumab', 'nivolumab']
    }
    # Add more as needed...
}

# Mutation signatures for different cancer types
MUTATION_SIGNATURES = {
    'smoking': {'C>A': 0.4, 'C>G': 0.2, 'C>T': 0.3, 'T>A': 0.1},
    'uv_exposure': {'C>T': 0.7, 'CC>TT': 0.2, 'C>A': 0.1},
    'aging': {'C>T': 0.6, 'T>C': 0.2, 'C>G': 0.1, 'T>A': 0.1},
    'dna_repair_deficiency': {'indels': 0.4, 'C>T': 0.3, 'T>G': 0.2, 'C>A': 0.1},
    'chemotherapy': {'T>A': 0.4, 'T>C': 0.3, 'C>A': 0.2, 'C>T': 0.1}
}

# Random seeds for reproducibility
RANDOM_SEEDS = {
    'numpy': 42,
    'torch': 42,
    'python': 42
}

# Set random seeds
np.random.seed(RANDOM_SEEDS['numpy'])
torch.manual_seed(RANDOM_SEEDS['torch'])
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEEDS['torch'])

print(f"Configuration loaded successfully. Using device: {DEVICE}")
