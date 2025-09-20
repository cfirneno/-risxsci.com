"""
Data Generation Module
=====================

Synthetic cancer data generation including SyntheticCancerDataset and 
realistic data simulation functions for training and testing.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict
import random
from scipy import stats
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import io
import base64

# Import from config module
try:
    from config import CancerType, TumorStage, GeneticMarker
except ImportError:
    # Fallback if config module not available
    from enum import Enum
    class CancerType(Enum):
        BREAST_DUCTAL = "breast_ductal"
        LUNG_ADENOCARCINOMA = "lung_adenocarcinoma"
        COLON_ADENOCARCINOMA = "colon_adenocarcinoma"
        PROSTATE_ADENOCARCINOMA = "prostate_adenocarcinoma"
        MELANOMA = "melanoma"
        PANCREATIC_ADENOCARCINOMA = "pancreatic_adenocarcinoma"

logger = logging.getLogger(__name__)

@dataclass
class PatientData:
    """Comprehensive patient data structure"""
    patient_id: str
    
    # Demographics
    age: float
    gender: str
    race: str
    ethnicity: str
    
    # Cancer characteristics
    cancer_type: str
    tumor_stage: str
    tumor_grade: int
    histology: str
    primary_site: str
    diagnosis_date: datetime
    
    # WSI data
    wsi_features: np.ndarray
    wsi_metadata: Dict[str, Any]
    
    # Genomics data
    mutations: Dict[str, Any]
    copy_number_variants: Dict[str, float]
    gene_expression: Dict[str, float]
    fusion_genes: List[str]
    
    # Methylation data
    cpg_methylation: Dict[str, float]
    global_methylation: float
    methylation_signature: str
    
    # Clinical data
    laboratory_values: Dict[str, float]
    biomarkers: Dict[str, float]
    performance_status: int
    comorbidities: List[str]
    
    # Treatment and outcomes
    treatments: List[Dict[str, Any]]
    response_data: Dict[str, Any]
    survival_months: float
    event_occurred: bool
    last_followup: datetime

@dataclass
class DataGenerationConfig:
    """Configuration for data generation"""
    num_samples: int = 1000
    cancer_distribution: Dict[str, float] = field(default_factory=dict)
    age_distribution: Tuple[float, float] = (45.0, 75.0)
    gender_distribution: Dict[str, float] = field(default_factory=lambda: {'male': 0.4, 'female': 0.6})
    mutation_rate: float = 0.15
    survival_censoring_rate: float = 0.3
    noise_level: float = 0.1
    correlation_strength: float = 0.7
    
class SyntheticCancerDataset(Dataset):
    """Comprehensive synthetic cancer dataset with realistic patterns"""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 mode: str = 'train',
                 config: Optional[DataGenerationConfig] = None,
                 seed: int = 42):
        
        self.num_samples = num_samples
        self.mode = mode
        self.config = config or DataGenerationConfig()
        self.seed = seed
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        
        # Cancer type information
        self.cancer_types = list(CancerType)
        self.cancer_prevalence = self._get_cancer_prevalence()
        self.gene_signatures = self._load_gene_signatures()
        self.mutation_profiles = self._load_mutation_profiles()
        
        # Generate dataset
        self.patients = self._generate_patient_cohort()
        
        logger.info(f"Generated {len(self.patients)} {mode} samples")
    
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single patient sample as tensors"""
        
        patient = self.patients[idx]
        
        # Convert to tensors
        sample = {
            'patient_id': patient.patient_id,
            'wsi_features': torch.FloatTensor(patient.wsi_features),
            'genomics_features': self._genomics_to_tensor(patient),
            'methylation_features': self._methylation_to_tensor(patient),
            'clinical_features': self._clinical_to_tensor(patient),
            'cancer_type': self._encode_cancer_type(patient.cancer_type),
            'tumor_stage': self._encode_tumor_stage(patient.tumor_stage),
            'survival_months': torch.FloatTensor([patient.survival_months]),
            'event_occurred': torch.FloatTensor([float(patient.event_occurred)]),
            'age': torch.FloatTensor([patient.age]),
            'gender': torch.FloatTensor([1.0 if patient.gender == 'male' else 0.0])
        }
        
        return sample
    
    def _get_cancer_prevalence(self) -> Dict[str, float]:
        """Get realistic cancer prevalence rates"""
        
        prevalence = {
            'breast_ductal': 0.15,
            'lung_adenocarcinoma': 0.12,
            'colon_adenocarcinoma': 0.10,
            'prostate_adenocarcinoma': 0.09,
            'melanoma': 0.06,
            'pancreatic_adenocarcinoma': 0.04,
            'ovarian_serous': 0.03,
            'glioblastoma': 0.02,
            'hepatocellular': 0.03,
            'renal_clear_cell': 0.03,
            'bladder_transitional': 0.04,
            'lymphoma_nhl': 0.05,
            'leukemia_aml': 0.03,
            'sarcoma_soft_tissue': 0.02,
            'thyroid_papillary': 0.04,
            'gastric_adenocarcinoma': 0.03,
            'esophageal_adenocarcinoma': 0.02,
            'head_neck_squamous': 0.04,
            'cervical_squamous': 0.02,
            'endometrial_adenocarcinoma': 0.03,
            'other': 0.12
        }
        
        return prevalence
    
    def _load_gene_signatures(self) -> Dict[str, Dict[str, float]]:
        """Load cancer-specific gene expression signatures"""
        
        signatures = {
            'breast_ductal': {
                'ESR1': 2.5, 'PGR': 2.0, 'ERBB2': 1.2, 'MKI67': 1.8,
                'BRCA1': 0.8, 'BRCA2': 0.9, 'TP53': 0.7, 'PIK3CA': 1.5,
                'CDH1': 2.0, 'KRT19': 2.2, 'GATA3': 2.3, 'FOXA1': 2.1
            },
            'lung_adenocarcinoma': {
                'EGFR': 2.0, 'KRAS': 1.5, 'ALK': 0.3, 'ROS1': 0.2,
                'TP53': 0.5, 'STK11': 0.8, 'KEAP1': 0.7, 'NF1': 0.9,
                'TTF1': 2.5, 'CK7': 2.0, 'SP-A': 1.8, 'SP-B': 1.6
            },
            'colon_adenocarcinoma': {
                'APC': 0.3, 'KRAS': 1.2, 'PIK3CA': 1.4, 'TP53': 0.4,
                'SMAD4': 0.8, 'BRAF': 1.1, 'MSH2': 1.0, 'MLH1': 0.9,
                'CDX2': 2.5, 'CK20': 2.3, 'CEA': 2.0, 'SATB2': 2.1
            },
            'prostate_adenocarcinoma': {
                'AR': 2.8, 'PSA': 3.0, 'TMPRSS2': 2.2, 'ERG': 1.5,
                'PTEN': 0.6, 'TP53': 0.8, 'RB1': 0.9, 'BRCA2': 0.8,
                'NKX3-1': 2.5, 'AMACR': 2.0, 'PSM': 1.8, 'PSAP': 2.2
            },
            'melanoma': {
                'BRAF': 1.8, 'NRAS': 1.3, 'KIT': 1.0, 'NF1': 0.7,
                'TP53': 0.6, 'CDKN2A': 0.5, 'PTEN': 0.8, 'PIK3CA': 1.2,
                'SOX10': 2.5, 'MITF': 2.2, 'S100': 2.8, 'MART1': 2.3
            },
            'pancreatic_adenocarcinoma': {
                'KRAS': 1.8, 'TP53': 0.3, 'CDKN2A': 0.4, 'SMAD4': 0.5,
                'BRCA1': 0.7, 'BRCA2': 0.6, 'ATM': 0.8, 'PALB2': 0.7,
                'CK19': 2.0, 'CA19-9': 2.5, 'CEA': 1.8, 'MUC1': 2.2
            }
        }
        
        # Add default signature for missing cancer types
        default_signature = {f'GENE_{i}': np.random.normal(1.0, 0.3) for i in range(50)}
        
        for cancer_type in self.cancer_types:
            cancer_name = cancer_type.value if hasattr(cancer_type, 'value') else str(cancer_type)
            if cancer_name not in signatures:
                signatures[cancer_name] = default_signature.copy()
        
        return signatures
    
    def _load_mutation_profiles(self) -> Dict[str, Dict[str, float]]:
        """Load cancer-specific mutation frequencies"""
        
        profiles = {
            'breast_ductal': {
                'PIK3CA': 0.36, 'TP53': 0.30, 'CDH1': 0.16, 'GATA3': 0.11,
                'MAP3K1': 0.10, 'MLL3': 0.09, 'MAP2K4': 0.08, 'NCOR1': 0.07
            },
            'lung_adenocarcinoma': {
                'KRAS': 0.32, 'TP53': 0.46, 'STK11': 0.17, 'EGFR': 0.14,
                'KEAP1': 0.16, 'NF1': 0.11, 'BRAF': 0.07, 'RBM10': 0.09
            },
            'colon_adenocarcinoma': {
                'APC': 0.81, 'TP53': 0.60, 'KRAS': 0.43, 'PIK3CA': 0.15,
                'SMAD4': 0.19, 'FBXW7': 0.11, 'NRAS': 0.05, 'BRAF': 0.10
            },
            'prostate_adenocarcinoma': {
                'TP53': 0.42, 'PTEN': 0.41, 'FOXA1': 0.14, 'SPOP': 0.11,
                'TMPRSS2-ERG': 0.46, 'CHD1': 0.07, 'RB1': 0.06, 'MYC': 0.12
            },
            'melanoma': {
                'BRAF': 0.45, 'NRAS': 0.28, 'NF1': 0.14, 'TP53': 0.15,
                'CDKN2A': 0.58, 'PTEN': 0.23, 'KIT': 0.02, 'GNA11': 0.03
            },
            'pancreatic_adenocarcinoma': {
                'KRAS': 0.95, 'TP53': 0.72, 'CDKN2A': 0.90, 'SMAD4': 0.55,
                'ARID1A': 0.12, 'TGFBR2': 0.07, 'GNAS': 0.06, 'RNF43': 0.05
            }
        }
        
        # Add default profile for missing cancer types
        default_profile = {f'GENE_{i}': np.random.beta(2, 8) for i in range(20)}
        
        for cancer_type in self.cancer_types:
            cancer_name = cancer_type.value if hasattr(cancer_type, 'value') else str(cancer_type)
            if cancer_name not in profiles:
                profiles[cancer_name] = default_profile.copy()
        
        return profiles
    
    def _generate_patient_cohort(self) -> List[PatientData]:
        """Generate a cohort of synthetic patients"""
        
        patients = []
        
        for i in range(self.num_samples):
            # Generate basic demographics
            age = self._sample_age()
            gender = self._sample_gender()
            race = self._sample_race()
            ethnicity = self._sample_ethnicity()
            
            # Sample cancer type
            cancer_type = self._sample_cancer_type()
            
            # Generate cancer characteristics
            tumor_stage = self._sample_tumor_stage(cancer_type, age)
            tumor_grade = self._sample_tumor_grade(cancer_type)
            histology = self._sample_histology(cancer_type)
            primary_site = self._sample_primary_site(cancer_type)
            
            # Generate diagnosis date
            diagnosis_date = self._sample_diagnosis_date()
            
            # Generate multimodal data
            wsi_features = self._generate_wsi_features(cancer_type, tumor_stage, tumor_grade)
            mutations = self._generate_mutations(cancer_type, age, gender)
            gene_expression = self._generate_gene_expression(cancer_type, mutations)
            methylation_data = self._generate_methylation_data(cancer_type, age)
            clinical_data = self._generate_clinical_data(cancer_type, age, gender, tumor_stage)
            
            # Generate outcomes
            survival_data = self._generate_survival_data(
                cancer_type, tumor_stage, age, mutations, clinical_data
            )
            
            # Create patient
            patient = PatientData(
                patient_id=f"SYNTH_{i:06d}",
                age=age,
                gender=gender,
                race=race,
                ethnicity=ethnicity,
                cancer_type=cancer_type,
                tumor_stage=tumor_stage,
                tumor_grade=tumor_grade,
                histology=histology,
                primary_site=primary_site,
                diagnosis_date=diagnosis_date,
                wsi_features=wsi_features['features'],
                wsi_metadata=wsi_features['metadata'],
                mutations=mutations,
                copy_number_variants=gene_expression['cnv'],
                gene_expression=gene_expression['expression'],
                fusion_genes=gene_expression['fusions'],
                cpg_methylation=methylation_data['cpg_sites'],
                global_methylation=methylation_data['global_methylation'],
                methylation_signature=methylation_data['signature'],
                laboratory_values=clinical_data['lab_values'],
                biomarkers=clinical_data['biomarkers'],
                performance_status=clinical_data['performance_status'],
                comorbidities=clinical_data['comorbidities'],
                treatments=clinical_data['treatments'],
                response_data=survival_data['response'],
                survival_months=survival_data['survival_months'],
                event_occurred=survival_data['event_occurred'],
                last_followup=survival_data['last_followup']
            )
            
            patients.append(patient)
        
        return patients
    
    def _sample_age(self) -> float:
        """Sample realistic age distribution"""
        # Bimodal distribution: younger and older patients
        if np.random.random() < 0.3:
            # Younger patients (30-50)
            age = np.random.gamma(2, 10) + 30
        else:
            # Older patients (50-85)
            age = np.random.gamma(3, 8) + 50
        
        return float(np.clip(age, 18, 100))
    
    def _sample_gender(self) -> str:
        """Sample gender distribution"""
        return np.random.choice(['male', 'female'], p=[0.45, 0.55])
    
    def _sample_race(self) -> str:
        """Sample race distribution"""
        races = ['white', 'black', 'asian', 'hispanic', 'other']
        probs = [0.65, 0.15, 0.08, 0.10, 0.02]
        return np.random.choice(races, p=probs)
    
    def _sample_ethnicity(self) -> str:
        """Sample ethnicity distribution"""
        ethnicities = ['non_hispanic', 'hispanic', 'unknown']
        probs = [0.80, 0.18, 0.02]
        return np.random.choice(ethnicities, p=probs)
    
    def _sample_cancer_type(self) -> str:
        """Sample cancer type based on prevalence"""
        cancer_names = list(self.cancer_prevalence.keys())
        probs = list(self.cancer_prevalence.values())
        return np.random.choice(cancer_names, p=probs)
    
    def _sample_tumor_stage(self, cancer_type: str, age: float) -> str:
        """Sample tumor stage with age-dependent probability"""
        stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        
        # Age-dependent staging (older patients more likely advanced)
        age_factor = (age - 40) / 40  # Normalized age factor
        
        # Base probabilities
        base_probs = [0.3, 0.35, 0.25, 0.10]
        
        # Adjust for age (older = more advanced)
        stage_shift = age_factor * 0.15
        probs = [
            base_probs[0] - stage_shift,
            base_probs[1] - stage_shift * 0.5,
            base_probs[2] + stage_shift * 0.5,
            base_probs[3] + stage_shift
        ]
        
        # Ensure probabilities sum to 1 and are positive
        probs = np.maximum(probs, 0.01)
        probs = probs / np.sum(probs)
        
        return np.random.choice(stages, p=probs)
    
    def _sample_tumor_grade(self, cancer_type: str) -> int:
        """Sample tumor grade (1-4)"""
        # Grade distribution varies by cancer type
        if 'breast' in cancer_type:
            probs = [0.20, 0.45, 0.30, 0.05]  # Mostly grade 2-3
        elif 'lung' in cancer_type:
            probs = [0.10, 0.25, 0.50, 0.15]  # Often higher grade
        else:
            probs = [0.15, 0.35, 0.35, 0.15]  # Balanced distribution
        
        return int(np.random.choice([1, 2, 3, 4], p=probs))
    
    def _sample_histology(self, cancer_type: str) -> str:
        """Sample histological subtype"""
        histology_map = {
            'breast_ductal': 'invasive_ductal_carcinoma',
            'lung_adenocarcinoma': 'adenocarcinoma',
            'colon_adenocarcinoma': 'adenocarcinoma',
            'prostate_adenocarcinoma': 'adenocarcinoma',
            'melanoma': 'malignant_melanoma',
            'pancreatic_adenocarcinoma': 'ductal_adenocarcinoma'
        }
        
        return histology_map.get(cancer_type, 'adenocarcinoma')
    
    def _sample_primary_site(self, cancer_type: str) -> str:
        """Sample primary tumor site"""
        site_map = {
            'breast_ductal': 'left_breast_upper_outer',
            'lung_adenocarcinoma': 'right_upper_lobe',
            'colon_adenocarcinoma': 'sigmoid_colon',
            'prostate_adenocarcinoma': 'prostate_peripheral_zone',
            'melanoma': 'skin_back',
            'pancreatic_adenocarcinoma': 'pancreatic_head'
        }
        
        return site_map.get(cancer_type, 'primary_site')
    
    def _sample_diagnosis_date(self) -> datetime:
        """Sample diagnosis date"""
        # Random date within last 5 years
        days_ago = np.random.randint(0, 1825)  # 5 years
        return datetime.now() - timedelta(days=days_ago)
    
    def _generate_wsi_features(self, cancer_type: str, stage: str, grade: int) -> Dict[str, Any]:
        """Generate WSI features with cancer-specific patterns"""
        
        # Base feature dimension
        feature_dim = 512
        
        # Generate features with cancer-specific patterns
        if 'breast' in cancer_type:
            # Breast-specific patterns
            base_features = np.random.normal(0.5, 0.2, feature_dim)
            # Add ductal patterns
            base_features[:50] += np.random.normal(0.3, 0.1, 50)
            
        elif 'lung' in cancer_type:
            # Lung-specific patterns
            base_features = np.random.normal(0.3, 0.25, feature_dim)
            # Add glandular patterns
            base_features[50:100] += np.random.normal(0.4, 0.15, 50)
            
        elif 'melanoma' in cancer_type:
            # Melanoma-specific patterns
            base_features = np.random.normal(0.2, 0.3, feature_dim)
            # Add pigmentation patterns
            base_features[100:150] += np.random.normal(0.6, 0.2, 50)
            
        else:
            # Generic pattern
            base_features = np.random.normal(0.4, 0.2, feature_dim)
        
        # Modify based on grade (higher grade = more irregular patterns)
        grade_factor = (grade - 1) / 3.0  # Normalize to 0-1
        noise_level = 0.1 + grade_factor * 0.2
        base_features += np.random.normal(0, noise_level, feature_dim)
        
        # Modify based on stage
        stage_num = int(stage.split()[-1]) if 'Stage' in stage else 1
        stage_factor = (stage_num - 1) / 3.0
        base_features += stage_factor * np.random.normal(0.1, 0.05, feature_dim)
        
        # Clip to reasonable range
        features = np.clip(base_features, 0, 1)
        
        metadata = {
            'tumor_cellularity': np.random.uniform(0.3, 0.9),
            'necrosis_percentage': np.random.uniform(0, 0.3),
            'inflammation_score': np.random.randint(0, 4),
            'vessel_density': np.random.uniform(0.1, 0.8),
            'nuclear_grade': grade,
            'mitotic_count': np.random.poisson(5 * grade),
            'ki67_index': np.random.uniform(0.1, 0.8)
        }
        
        return {'features': features, 'metadata': metadata}
    
    def _generate_mutations(self, cancer_type: str, age: float, gender: str) -> Dict[str, Any]:
        """Generate mutation data with realistic patterns"""
        
        mutations = {}
        
        # Get cancer-specific mutation profile
        if cancer_type in self.mutation_profiles:
            mutation_freqs = self.mutation_profiles[cancer_type]
        else:
            # Default mutation profile
            mutation_freqs = {f'GENE_{i}': np.random.beta(1, 10) for i in range(20)}
        
        # Generate mutations based on frequencies
        for gene, freq in mutation_freqs.items():
            # Age-dependent mutation accumulation
            age_factor = 1 + (age - 50) / 100  # Slight increase with age
            adjusted_freq = min(freq * age_factor, 0.95)
            
            if np.random.random() < adjusted_freq:
                # Generate mutation details
                mutation_types = ['missense', 'nonsense', 'frameshift', 'splice_site', 'in_frame_indel']
                mut_type = np.random.choice(mutation_types, p=[0.5, 0.2, 0.15, 0.1, 0.05])
                
                mutations[gene] = {
                    'mutation_type': mut_type,
                    'variant_allele_frequency': np.random.uniform(0.1, 0.9),
                    'pathogenicity': np.random.choice(['pathogenic', 'likely_pathogenic', 'uncertain', 'benign'], 
                                                    p=[0.3, 0.2, 0.3, 0.2]),
                    'coverage': np.random.randint(50, 500),
                    'quality_score': np.random.uniform(20, 60)
                }
        
        return mutations
    
    def _generate_gene_expression(self, cancer_type: str, mutations: Dict) -> Dict[str, Any]:
        """Generate gene expression with mutation-dependent patterns"""
        
        # Get cancer-specific signature
        if cancer_type in self.gene_signatures:
            signature = self.gene_signatures[cancer_type]
        else:
            signature = {f'GENE_{i}': np.random.lognormal(0, 0.5) for i in range(100)}
        
        expression = {}
        
        # Generate expression levels
        for gene, base_expr in signature.items():
            # Start with base expression
            expr_level = base_expr
            
            # Modify based on mutations
            if gene in mutations:
                mut_type = mutations[gene]['mutation_type']
                if mut_type in ['nonsense', 'frameshift']:
                    # Loss of function mutations reduce expression
                    expr_level *= np.random.uniform(0.1, 0.7)
                elif mut_type == 'missense':
                    # Variable effect
                    expr_level *= np.random.uniform(0.5, 1.5)
            
            # Add noise
            expr_level *= np.random.lognormal(0, 0.2)
            expression[gene] = max(expr_level, 0.01)  # Minimum expression
        
        # Generate copy number variants
        cnv = {}
        cnv_genes = np.random.choice(list(expression.keys()), size=20, replace=False)
        for gene in cnv_genes:
            cnv_value = np.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])  # Deletion, neutral, amplification
            cnv[gene] = cnv_value
            
            # Modify expression based on CNV
            if cnv_value == -1:  # Deletion
                expression[gene] *= np.random.uniform(0.3, 0.8)
            elif cnv_value == 1:  # Amplification
                expression[gene] *= np.random.uniform(1.2, 3.0)
        
        # Generate fusion genes (rare events)
        fusion_genes = []
        if np.random.random() < 0.05:  # 5% chance of fusion
            gene_list = list(expression.keys())
            gene1 = np.random.choice(gene_list)
            gene2 = np.random.choice([g for g in gene_list if g != gene1])
            fusion_genes.append(f"{gene1}-{gene2}")
        
        return {
            'expression': expression,
            'cnv': cnv,
            'fusions': fusion_genes
        }
    
    def _generate_methylation_data(self, cancer_type: str, age: float) -> Dict[str, Any]:
        """Generate methylation data with age and cancer-specific patterns"""
        
        # Number of CpG sites to simulate
        num_cpg_sites = 1000
        
        # Age-dependent global methylation
        age_factor = (age - 30) / 50  # Normalized age
        global_methylation = 0.4 + age_factor * 0.3 + np.random.normal(0, 0.1)
        global_methylation = np.clip(global_methylation, 0.2, 0.9)
        
        # Generate CpG site methylation
        cpg_methylation = {}
        
        for i in range(num_cpg_sites):
            site_name = f"cg{i:08d}"
            
            # Base methylation level
            base_meth = global_methylation + np.random.normal(0, 0.2)
            
            # Cancer-specific alterations
            if cancer_type in ['colon_adenocarcinoma', 'glioblastoma']:
                # CIMP phenotype - hypermethylation
                if i < 200:  # First 200 sites are hypermethylated
                    base_meth += np.random.uniform(0.2, 0.5)
            
            if 'breast' in cancer_type:
                # Breast-specific methylation patterns
                if i >= 500 and i < 700:  # Middle region hypomethylated
                    base_meth -= np.random.uniform(0.1, 0.3)
            
            # Clip to valid range
            cpg_methylation[site_name] = np.clip(base_meth, 0.0, 1.0)
        
        # Determine methylation signature
        hypermethylated_sites = sum(1 for v in cpg_methylation.values() if v > 0.7)
        if hypermethylated_sites > 200:
            signature = 'CIMP_high'
        elif hypermethylated_sites > 100:
            signature = 'CIMP_low'
        else:
            signature = 'CIMP_negative'
        
        return {
            'cpg_sites': cpg_methylation,
            'global_methylation': global_methylation,
            'signature': signature,
            'hypermethylated_count': hypermethylated_sites
        }
    
    def _generate_clinical_data(self, cancer_type: str, age: float, gender: str, stage: str) -> Dict[str, Any]:
        """Generate clinical data and laboratory values"""
        
        # Laboratory values
        lab_values = {
            'hemoglobin': np.random.normal(13.0 if gender == 'male' else 12.0, 1.5),
            'white_blood_cells': np.random.lognormal(np.log(7000), 0.3),
            'platelets': np.random.normal(250000, 50000),
            'creatinine': np.random.normal(1.0, 0.2),
            'albumin': np.random.normal(4.0, 0.5),
            'total_bilirubin': np.random.lognormal(np.log(0.8), 0.3),
            'alt': np.random.lognormal(np.log(25), 0.4),
            'ast': np.random.lognormal(np.log(25), 0.4),
            'ldh': np.random.lognormal(np.log(200), 0.3)
        }
        
        # Cancer-specific biomarkers
        biomarkers = {}
        
        if 'breast' in cancer_type:
            biomarkers['CEA'] = np.random.lognormal(np.log(2.5), 0.8)
            biomarkers['CA15_3'] = np.random.lognormal(np.log(20), 0.7)
            biomarkers['CA27_29'] = np.random.lognormal(np.log(25), 0.6)
            
        elif 'lung' in cancer_type:
            biomarkers['CEA'] = np.random.lognormal(np.log(5.0), 0.9)
            biomarkers['CYFRA21_1'] = np.random.lognormal(np.log(3.0), 0.8)
            biomarkers['NSE'] = np.random.lognormal(np.log(15), 0.7)
            
        elif 'colon' in cancer_type:
            biomarkers['CEA'] = np.random.lognormal(np.log(8.0), 1.2)
            biomarkers['CA19_9'] = np.random.lognormal(np.log(50), 1.0)
            
        elif 'prostate' in cancer_type:
            # Age-dependent PSA
            base_psa = 1.0 + (age - 50) * 0.1
            biomarkers['PSA'] = np.random.lognormal(np.log(base_psa), 0.8)
            biomarkers['free_PSA_ratio'] = np.random.uniform(0.1, 0.3)
            
        # Performance status (ECOG)
        stage_num = int(stage.split()[-1]) if 'Stage' in stage else 1
        perf_probs = {
            1: [0.6, 0.3, 0.1, 0.0, 0.0],  # Stage I
            2: [0.4, 0.4, 0.15, 0.05, 0.0],  # Stage II
            3: [0.2, 0.4, 0.25, 0.1, 0.05],  # Stage III
            4: [0.1, 0.3, 0.3, 0.2, 0.1]   # Stage IV
        }
        performance_status = np.random.choice([0, 1, 2, 3, 4], p=perf_probs.get(stage_num, [0.4, 0.3, 0.2, 0.1, 0.0]))
        
        # Comorbidities (age-dependent)
        possible_comorbidities = [
            'hypertension', 'diabetes', 'heart_disease', 'copd', 'kidney_disease',
            'liver_disease', 'previous_cancer', 'depression', 'osteoporosis'
        ]
        
        num_comorbidities = np.random.poisson(max(0, (age - 40) / 20))
        comorbidities = list(np.random.choice(possible_comorbidities, 
                                            size=min(num_comorbidities, len(possible_comorbidities)), 
                                            replace=False))
        
        # Treatment history
        treatments = []
        if stage_num >= 2:  # Stages II-IV get treatment
            treatment_types = ['surgery', 'chemotherapy', 'radiation', 'immunotherapy', 'targeted_therapy']
            num_treatments = np.random.randint(1, 4)
            selected_treatments = np.random.choice(treatment_types, size=num_treatments, replace=False)
            
            for treatment in selected_treatments:
                treatments.append({
                    'type': treatment,
                    'start_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                    'duration_days': np.random.randint(7, 180),
                    'response': np.random.choice(['complete_response', 'partial_response', 'stable_disease', 'progressive_disease'],
                                               p=[0.15, 0.35, 0.35, 0.15])
                })
        
        return {
            'lab_values': lab_values,
            'biomarkers': biomarkers,
            'performance_status': performance_status,
            'comorbidities': comorbidities,
            'treatments': treatments
        }
    
    def _generate_survival_data(self, cancer_type: str, stage: str, age: float, 
                              mutations: Dict, clinical_data: Dict) -> Dict[str, Any]:
        """Generate survival outcomes with realistic patterns"""
        
        # Base survival based on cancer type and stage
        stage_num = int(stage.split()[-1]) if 'Stage' in stage else 1
        
        # Stage-specific median survival (months)
        stage_survival = {
            1: 120,  # 10 years
            2: 84,   # 7 years
            3: 48,   # 4 years
            4: 18    # 1.5 years
        }
        
        median_survival = stage_survival.get(stage_num, 60)
        
        # Cancer-specific adjustments
        cancer_factors = {
            'pancreatic_adenocarcinoma': 0.3,  # Very poor prognosis
            'glioblastoma': 0.4,              # Poor prognosis
            'lung_adenocarcinoma': 0.6,       # Moderate prognosis
            'breast_ductal': 1.4,             # Better prognosis
            'prostate_adenocarcinoma': 1.8,   # Good prognosis
            'melanoma': 1.2                   # Variable prognosis
        }
        
        cancer_factor = cancer_factors.get(cancer_type, 1.0)
        median_survival *= cancer_factor
        
        # Age adjustment (older = worse prognosis)
        age_factor = max(0.5, 1.0 - (age - 60) / 100)
        median_survival *= age_factor
        
        # Performance status adjustment
        ps_factor = max(0.3, 1.0 - clinical_data['performance_status'] * 0.2)
        median_survival *= ps_factor
        
        # Mutation-based adjustments
        if 'TP53' in mutations:
            median_survival *= 0.8  # TP53 mutations worsen prognosis
        if 'BRCA1' in mutations or 'BRCA2' in mutations:
            median_survival *= 1.3  # BRCA mutations can improve response to treatment
        
        # Generate actual survival time (Weibull distribution)
        shape = 1.5
        scale = median_survival / np.log(2) ** (1/shape)
        survival_months = np.random.weibull(shape) * scale
        
        # Censoring (some patients are still alive)
        censoring_rate = self.config.survival_censoring_rate
        event_occurred = np.random.random() > censoring_rate
        
        if not event_occurred:
            # If censored, observed time is shorter
            survival_months = np.random.uniform(0.5, 1.0) * survival_months
        
        # Last follow-up date
        follow_up_days = min(survival_months * 30, 1825)  # Max 5 years
        last_followup = datetime.now() - timedelta(days=follow_up_days)
        
        # Treatment response data
        if clinical_data['treatments']:
            best_response = np.random.choice(
                ['complete_response', 'partial_response', 'stable_disease', 'progressive_disease'],
                p=[0.1, 0.3, 0.4, 0.2]
            )
        else:
            best_response = 'not_evaluable'
        
        response_data = {
            'best_response': best_response,
            'time_to_progression': np.random.uniform(3, 24) if best_response != 'progressive_disease' else np.random.uniform(1, 6),
            'response_duration': np.random.uniform(6, 36) if best_response in ['complete_response', 'partial_response'] else 0
        }
        
        return {
            'survival_months': float(survival_months),
            'event_occurred': event_occurred,
            'last_followup': last_followup,
            'response': response_data
        }
    
    def _genomics_to_tensor(self, patient: PatientData) -> torch.Tensor:
        """Convert genomics data to tensor"""
        
        # Create feature vector
        features = []
        
        # Mutation features (binary indicators for top 100 genes)
        top_genes = ['TP53', 'KRAS', 'PIK3CA', 'EGFR', 'PTEN', 'BRAF', 'APC', 'BRCA1', 'BRCA2', 'MYC'] * 10  # 100 genes
        for gene in top_genes:
            features.append(1.0 if gene in patient.mutations else 0.0)
        
        # Expression features (top 50 expressed genes)
        expr_genes = list(patient.gene_expression.keys())[:50]
        for gene in expr_genes:
            features.append(np.log1p(patient.gene_expression.get(gene, 0.01)))
        
        # CNV features
        cnv_genes = list(patient.copy_number_variants.keys())[:20]
        for gene in cnv_genes:
            features.append(float(patient.copy_number_variants.get(gene, 0)))
        
        # Pad or truncate to fixed size (2048)
        target_size = 2048
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return torch.FloatTensor(features)
    
    def _methylation_to_tensor(self, patient: PatientData) -> torch.Tensor:
        """Convert methylation data to tensor"""
        
        # Get methylation values for top CpG sites
        cpg_sites = list(patient.cpg_methylation.keys())[:1000]
        features = [patient.cpg_methylation[site] for site in cpg_sites]
        
        # Add global methylation
        features.append(patient.global_methylation)
        
        # Add signature encoding
        signature_encoding = {
            'CIMP_high': [1, 0, 0],
            'CIMP_low': [0, 1, 0],
            'CIMP_negative': [0, 0, 1]
        }
        features.extend(signature_encoding.get(patient.methylation_signature, [0, 0, 0]))
        
        return torch.FloatTensor(features)
    
    def _clinical_to_tensor(self, patient: PatientData) -> torch.Tensor:
        """Convert clinical data to tensor"""
        
        features = []
        
        # Demographics
        features.append(patient.age / 100.0)  # Normalized age
        features.append(1.0 if patient.gender == 'male' else 0.0)
        
        # Laboratory values
        lab_keys = ['hemoglobin', 'white_blood_cells', 'platelets', 'creatinine', 'albumin']
        for key in lab_keys:
            features.append(patient.laboratory_values.get(key, 0))
        
        # Biomarkers (log-transformed)
        for biomarker, value in patient.biomarkers.items():
            features.append(np.log1p(value))
        
        # Performance status
        features.append(patient.performance_status / 4.0)
        
        # Comorbidity count
        features.append(len(patient.comorbidities) / 10.0)
        
        # Treatment count
        features.append(len(patient.treatments) / 5.0)
        
        # Pad to fixed size (156)
        target_size = 156
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
        
        return torch.FloatTensor(features)
    
    def _encode_cancer_type(self, cancer_type: str) -> torch.Tensor:
        """Encode cancer type as one-hot vector"""
        cancer_names = list(self.cancer_prevalence.keys())
        encoding = [0.0] * len(cancer_names)
        
        if cancer_type in cancer_names:
            idx = cancer_names.index(cancer_type)
            encoding[idx] = 1.0
        
        return torch.FloatTensor(encoding)
    
    def _encode_tumor_stage(self, stage: str) -> torch.Tensor:
        """Encode tumor stage as one-hot vector"""
        stages = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
        encoding = [0.0] * len(stages)
        
        if stage in stages:
            idx = stages.index(stage)
            encoding[idx] = 1.0
        
        return torch.FloatTensor(encoding)

def create_synthetic_datasets(train_size: int = 1000, 
                            val_size: int = 200, 
                            test_size: int = 200,
                            config: Optional[DataGenerationConfig] = None) -> Tuple[SyntheticCancerDataset, SyntheticCancerDataset, SyntheticCancerDataset]:
    """Create train, validation, and test datasets"""
    
    if config is None:
        config = DataGenerationConfig()
    
    train_dataset = SyntheticCancerDataset(train_size, mode='train', config=config, seed=42)
    val_dataset = SyntheticCancerDataset(val_size, mode='val', config=config, seed=43)
    test_dataset = SyntheticCancerDataset(test_size, mode='test', config=config, seed=44)
    
    return train_dataset, val_dataset, test_dataset

def run_data_generation_demo():
    """Run a demonstration of data generation"""
    
    print("=" * 60)
    print("SYNTHETIC CANCER DATA GENERATION DEMONSTRATION")
    print("=" * 60)
    
    # Configuration
    config = DataGenerationConfig(
        num_samples=50,
        mutation_rate=0.15,
        survival_censoring_rate=0.3,
        noise_level=0.1
    )
    
    # Create dataset
    dataset = SyntheticCancerDataset(num_samples=10, config=config)
    
    print("DATASET OVERVIEW:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Cancer types: {len(dataset.cancer_prevalence)}")
    print(f"Mutation profiles: {len(dataset.mutation_profiles)}")
    print(f"Gene signatures: {len(dataset.gene_signatures)}")
    print()
    
    # Analyze first patient
    patient = dataset.patients[0]
    
    print("SAMPLE PATIENT DATA:")
    print(f"Patient ID: {patient.patient_id}")
    print(f"Age: {patient.age:.1f} years")
    print(f"Gender: {patient.gender}")
    print(f"Cancer Type: {patient.cancer_type}")
    print(f"Stage: {patient.tumor_stage}")
    print(f"Grade: {patient.tumor_grade}")
    print(f"Survival: {patient.survival_months:.1f} months")
    print(f"Event Occurred: {patient.event_occurred}")
    print()
    
    print("MULTIMODAL DATA:")
    print(f"WSI Features: {patient.wsi_features.shape}")
    print(f"Mutations: {len(patient.mutations)} genes")
    print(f"Gene Expression: {len(patient.gene_expression)} genes")
    print(f"Methylation Sites: {len(patient.cpg_methylation)} CpG sites")
    print(f"Lab Values: {len(patient.laboratory_values)} tests")
    print(f"Biomarkers: {len(patient.biomarkers)} markers")
    print(f"Treatments: {len(patient.treatments)} treatments")
    print()
    
    # Get tensor representation
    sample = dataset[0]
    
    print("TENSOR REPRESENTATIONS:")
    for key, tensor in sample.items():
        if isinstance(tensor, torch.Tensor):
            print(f"{key}: {tensor.shape}")
    print()
    
    # Dataset statistics
    print("DATASET STATISTICS:")
    
    # Cancer type distribution
    cancer_counts = defaultdict(int)
    ages = []
    survivals = []
    
    for patient in dataset.patients:
        cancer_counts[patient.cancer_type] += 1
        ages.append(patient.age)
        survivals.append(patient.survival_months)
    
    print("Cancer Type Distribution:")
    for cancer_type, count in sorted(cancer_counts.items()):
        print(f"  {cancer_type}: {count} ({count/len(dataset)*100:.1f}%)")
    
    print(f"\nAge Statistics:")
    print(f"  Mean: {np.mean(ages):.1f} years")
    print(f"  Std: {np.std(ages):.1f} years")
    print(f"  Range: {np.min(ages):.1f} - {np.max(ages):.1f} years")
    
    print(f"\nSurvival Statistics:")
    print(f"  Mean: {np.mean(survivals):.1f} months")
    print(f"  Median: {np.median(survivals):.1f} months")
    print(f"  Range: {np.min(survivals):.1f} - {np.max(survivals):.1f} months")
    
    print()
    
    # Data quality checks
    print("DATA QUALITY CHECKS:")
    
    # Check for missing values
    missing_data = False
    for i, patient in enumerate(dataset.patients[:5]):
        if not patient.wsi_features.any():
            print(f"  WARNING: Patient {i} has empty WSI features")
            missing_data = True
        if not patient.gene_expression:
            print(f"  WARNING: Patient {i} has no gene expression data")
            missing_data = True
    
    if not missing_data:
        print("  ✓ No missing data detected")
    
    # Check data ranges
    wsi_min = min(patient.wsi_features.min() for patient in dataset.patients)
    wsi_max = max(patient.wsi_features.max() for patient in dataset.patients)
    print(f"  WSI feature range: [{wsi_min:.3f}, {wsi_max:.3f}]")
    
    meth_values = [v for patient in dataset.patients for v in patient.cpg_methylation.values()]
    print(f"  Methylation range: [{min(meth_values):.3f}, {max(meth_values):.3f}]")
    
    print()
    print("SYNTHETIC DATA GENERATION DEMO COMPLETED SUCCESSFULLY!")
    
    return dataset

if __name__ == "__main__":
    # Run demonstration
    demo_dataset = run_data_generation_demo()
