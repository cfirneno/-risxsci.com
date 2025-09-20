"""
Clinical Integration Module
==========================

Comprehensive clinical data integration including biomarker processing,
treatment response prediction, and clinical decision support.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import re

# Import from config module
try:
    from config import *
except ImportError:
    # Fallback definitions if config module not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

@dataclass
class ClinicalData:
    """Comprehensive clinical data structure"""
    patient_id: str
    
    # Demographics
    age: float
    gender: str
    race: str
    ethnicity: str
    
    # Disease characteristics
    cancer_type: str
    tumor_stage: str
    tumor_grade: int
    histology: str
    primary_site: str
    diagnosis_date: datetime
    
    # Clinical measurements
    height_cm: float
    weight_kg: float
    bmi: float
    performance_status: int  # ECOG 0-4
    
    # Laboratory values
    laboratory_results: Dict[str, Dict[str, Any]]  # {test: {value, date, unit, ref_range}}
    
    # Biomarkers
    biomarkers: Dict[str, Dict[str, Any]]  # {marker: {value, method, date}}
    
    # Treatment history
    treatments: List[Dict[str, Any]]  # List of treatment records
    
    # Imaging studies
    imaging_studies: List[Dict[str, Any]]  # List of imaging records
    
    # Pathology reports
    pathology_reports: List[Dict[str, Any]]  # List of pathology records
    
    # Outcomes
    response_assessments: List[Dict[str, Any]]  # Treatment responses
    survival_status: str  # alive, deceased, unknown
    last_followup_date: datetime
    
    # Comorbidities
    comorbidities: List[str]
    medications: List[Dict[str, Any]]
    allergies: List[str]
    
    # Family history
    family_history: List[Dict[str, Any]]
    
    # Social history
    smoking_status: str
    alcohol_use: str
    drug_use: str
    occupation: str

@dataclass
class BiomarkerProfile:
    """Comprehensive biomarker profile"""
    
    # Protein biomarkers
    protein_markers: Dict[str, float] = field(default_factory=dict)
    
    # Genetic biomarkers
    genetic_markers: Dict[str, str] = field(default_factory=dict)
    
    # Metabolic biomarkers
    metabolic_markers: Dict[str, float] = field(default_factory=dict)
    
    # Inflammatory markers
    inflammatory_markers: Dict[str, float] = field(default_factory=dict)
    
    # Tumor markers
    tumor_markers: Dict[str, float] = field(default_factory=dict)
    
    # Hormone levels
    hormone_levels: Dict[str, float] = field(default_factory=dict)
    
    # Immune markers
    immune_markers: Dict[str, float] = field(default_factory=dict)
    
    # Coagulation markers
    coagulation_markers: Dict[str, float] = field(default_factory=dict)

@dataclass
class TreatmentResponse:
    """Treatment response assessment"""
    assessment_date: datetime
    treatment_regimen: str
    response_criteria: str  # RECIST, WHO, etc.
    response_category: str  # CR, PR, SD, PD
    response_percentage: float
    lesion_measurements: Dict[str, float]
    new_lesions: bool
    clinical_benefit: bool
    toxicity_grade: int  # 0-5 CTCAE
    quality_of_life_score: float
    biomarker_response: Dict[str, float]
    next_assessment_date: datetime

class ClinicalDataEncoder(nn.Module):
    """Neural network encoder for clinical data"""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Demographic encoder
        self.demographic_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Laboratory values encoder
        self.lab_encoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Biomarker encoder
        self.biomarker_encoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # Treatment history encoder
        self.treatment_encoder = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Disease characteristics encoder
        self.disease_encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # Combine all features
        combined_dim = 64 + 128 + 128 + 64 + 64  # 448
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, demographic_data: torch.Tensor,
                lab_data: torch.Tensor,
                biomarker_data: torch.Tensor,
                treatment_data: torch.Tensor,
                disease_data: torch.Tensor) -> torch.Tensor:
        
        # Encode each data type
        demo_features = self.demographic_encoder(demographic_data)
        lab_features = self.lab_encoder(lab_data)
        biomarker_features = self.biomarker_encoder(biomarker_data)
        treatment_features = self.treatment_encoder(treatment_data)
        disease_features = self.disease_encoder(disease_data)
        
        # Concatenate all features
        combined_features = torch.cat([
            demo_features,
            lab_features,
            biomarker_features,
            treatment_features,
            disease_features
        ], dim=1)
        
        # Final feature fusion
        encoded_features = self.feature_fusion(combined_features)
        
        return encoded_features

class TreatmentResponsePredictor(nn.Module):
    """Predict treatment response from clinical data"""
    
    def __init__(self, clinical_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        
        self.clinical_dim = clinical_dim
        self.hidden_dim = hidden_dim
        
        # Clinical feature processor
        self.clinical_processor = nn.Sequential(
            nn.Linear(clinical_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3)
        )
        
        # Response probability predictor
        self.response_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # CR, PR, SD, PD
            nn.Softmax(dim=1)
        )
        
        # Toxicity predictor
        self.toxicity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 6),  # Grade 0-5 toxicity
            nn.Softmax(dim=1)
        )
        
        # Survival benefit predictor
        self.survival_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability of survival benefit
        )
        
        # Quality of life predictor
        self.qol_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # QoL score 0-1
        )
    
    def forward(self, clinical_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Process clinical features
        processed_features = self.clinical_processor(clinical_features)
        
        # Make predictions
        response_probs = self.response_predictor(processed_features)
        toxicity_probs = self.toxicity_predictor(processed_features)
        survival_benefit = self.survival_predictor(processed_features)
        qol_score = self.qol_predictor(processed_features)
        
        return {
            'response_probabilities': response_probs,
            'toxicity_probabilities': toxicity_probs,
            'survival_benefit': survival_benefit,
            'quality_of_life': qol_score,
            'processed_features': processed_features
        }

class BiomarkerProcessor:
    """Process and analyze biomarker data"""
    
    def __init__(self):
        self.reference_ranges = self._load_reference_ranges()
        self.biomarker_weights = self._load_biomarker_weights()
        self.scalers = {}
        
    def _load_reference_ranges(self) -> Dict[str, Dict[str, float]]:
        """Load reference ranges for biomarkers"""
        
        return {
            # Tumor markers
            'CEA': {'normal_max': 5.0, 'unit': 'ng/mL'},
            'CA19-9': {'normal_max': 37.0, 'unit': 'U/mL'},
            'CA125': {'normal_max': 35.0, 'unit': 'U/mL'},
            'PSA': {'normal_max': 4.0, 'unit': 'ng/mL'},
            'AFP': {'normal_max': 10.0, 'unit': 'ng/mL'},
            'Beta-HCG': {'normal_max': 5.0, 'unit': 'mIU/mL'},
            
            # Inflammatory markers
            'CRP': {'normal_max': 3.0, 'unit': 'mg/L'},
            'ESR': {'normal_max': 30.0, 'unit': 'mm/hr'},
            'IL-6': {'normal_max': 7.0, 'unit': 'pg/mL'},
            'TNF-alpha': {'normal_max': 15.0, 'unit': 'pg/mL'},
            
            # Metabolic markers
            'LDH': {'normal_max': 250.0, 'unit': 'U/L'},
            'Albumin': {'normal_min': 3.5, 'normal_max': 5.0, 'unit': 'g/dL'},
            'Total_protein': {'normal_min': 6.0, 'normal_max': 8.3, 'unit': 'g/dL'},
            
            # Hematologic markers
            'Hemoglobin': {'normal_min': 12.0, 'normal_max': 16.0, 'unit': 'g/dL'},
            'WBC': {'normal_min': 4000, 'normal_max': 11000, 'unit': '/μL'},
            'Platelets': {'normal_min': 150000, 'normal_max': 450000, 'unit': '/μL'},
            'Neutrophils': {'normal_min': 2000, 'normal_max': 7500, 'unit': '/μL'},
            'Lymphocytes': {'normal_min': 1000, 'normal_max': 4000, 'unit': '/μL'},
            
            # Liver function
            'ALT': {'normal_max': 40.0, 'unit': 'U/L'},
            'AST': {'normal_max': 40.0, 'unit': 'U/L'},
            'Bilirubin': {'normal_max': 1.2, 'unit': 'mg/dL'},
            'ALP': {'normal_max': 120.0, 'unit': 'U/L'},
            
            # Kidney function
            'Creatinine': {'normal_max': 1.2, 'unit': 'mg/dL'},
            'BUN': {'normal_max': 20.0, 'unit': 'mg/dL'},
            'eGFR': {'normal_min': 90.0, 'unit': 'mL/min/1.73m²'},
            
            # Coagulation
            'PT': {'normal_max': 13.0, 'unit': 'seconds'},
            'PTT': {'normal_max': 35.0, 'unit': 'seconds'},
            'INR': {'normal_max': 1.1, 'unit': 'ratio'},
            
            # Cardiac markers
            'Troponin': {'normal_max': 0.04, 'unit': 'ng/mL'},
            'BNP': {'normal_max': 100.0, 'unit': 'pg/mL'},
            
            # Hormones
            'TSH': {'normal_min': 0.4, 'normal_max': 4.0, 'unit': 'mIU/L'},
            'T4': {'normal_min': 4.5, 'normal_max': 12.0, 'unit': 'μg/dL'},
            'Cortisol': {'normal_min': 6.0, 'normal_max': 23.0, 'unit': 'μg/dL'},
        }
    
    def _load_biomarker_weights(self) -> Dict[str, float]:
        """Load prognostic weights for different biomarkers"""
        
        return {
            # High prognostic value
            'CEA': 0.9,
            'CA19-9': 0.9,
            'PSA': 0.9,
            'LDH': 0.8,
            'Albumin': 0.8,
            'Hemoglobin': 0.7,
            
            # Moderate prognostic value
            'CRP': 0.6,
            'ESR': 0.5,
            'WBC': 0.6,
            'Platelets': 0.5,
            'ALT': 0.4,
            'AST': 0.4,
            'Creatinine': 0.6,
            
            # Lower prognostic value
            'Total_protein': 0.3,
            'BUN': 0.3,
            'PT': 0.3,
            'PTT': 0.3
        }
    
    def process_biomarkers(self, biomarker_data: Dict[str, float]) -> BiomarkerProfile:
        """Process raw biomarker data into structured profile"""
        
        profile = BiomarkerProfile()
        
        # Categorize biomarkers
        for marker, value in biomarker_data.items():
            if marker in ['CEA', 'CA19-9', 'CA125', 'PSA', 'AFP', 'Beta-HCG']:
                profile.tumor_markers[marker] = value
            elif marker in ['CRP', 'ESR', 'IL-6', 'TNF-alpha']:
                profile.inflammatory_markers[marker] = value
            elif marker in ['LDH', 'Albumin', 'Total_protein']:
                profile.metabolic_markers[marker] = value
            elif marker in ['Hemoglobin', 'WBC', 'Platelets', 'Neutrophils', 'Lymphocytes']:
                profile.protein_markers[marker] = value  # Blood proteins
            elif marker in ['TSH', 'T4', 'Cortisol']:
                profile.hormone_levels[marker] = value
            elif marker in ['PT', 'PTT', 'INR']:
                profile.coagulation_markers[marker] = value
            else:
                profile.protein_markers[marker] = value  # Default category
        
        return profile
    
    def calculate_biomarker_risk_score(self, profile: BiomarkerProfile) -> Dict[str, float]:
        """Calculate risk scores based on biomarker profile"""
        
        risk_scores = {}
        
        # Tumor marker risk
        tumor_risk = 0.0
        for marker, value in profile.tumor_markers.items():
            if marker in self.reference_ranges:
                normal_max = self.reference_ranges[marker].get('normal_max', float('inf'))
                if value > normal_max:
                    risk_contribution = min((value / normal_max - 1.0), 2.0)  # Cap at 2x normal
                    weight = self.biomarker_weights.get(marker, 0.5)
                    tumor_risk += risk_contribution * weight
        
        risk_scores['tumor_markers'] = min(tumor_risk, 1.0)
        
        # Inflammatory risk
        inflammatory_risk = 0.0
        for marker, value in profile.inflammatory_markers.items():
            if marker in self.reference_ranges:
                normal_max = self.reference_ranges[marker].get('normal_max', float('inf'))
                if value > normal_max:
                    risk_contribution = min((value / normal_max - 1.0), 2.0)
                    weight = self.biomarker_weights.get(marker, 0.5)
                    inflammatory_risk += risk_contribution * weight
        
        risk_scores['inflammatory'] = min(inflammatory_risk, 1.0)
        
        # Metabolic risk
        metabolic_risk = 0.0
        for marker, value in profile.metabolic_markers.items():
            if marker in self.reference_ranges:
                ref_range = self.reference_ranges[marker]
                if 'normal_max' in ref_range and value > ref_range['normal_max']:
                    risk_contribution = min((value / ref_range['normal_max'] - 1.0), 2.0)
                    weight = self.biomarker_weights.get(marker, 0.5)
                    metabolic_risk += risk_contribution * weight
                elif 'normal_min' in ref_range and value < ref_range['normal_min']:
                    risk_contribution = min((ref_range['normal_min'] / value - 1.0), 2.0)
                    weight = self.biomarker_weights.get(marker, 0.5)
                    metabolic_risk += risk_contribution * weight
        
        risk_scores['metabolic'] = min(metabolic_risk, 1.0)
        
        # Overall biomarker risk (weighted combination)
        overall_risk = (
            risk_scores['tumor_markers'] * 0.5 +
            risk_scores['inflammatory'] * 0.3 +
            risk_scores['metabolic'] * 0.2
        )
        
        risk_scores['overall'] = overall_risk
        
        return risk_scores
    
    def identify_abnormal_biomarkers(self, profile: BiomarkerProfile) -> Dict[str, List[Dict]]:
        """Identify biomarkers outside normal ranges"""
        
        abnormal = {
            'elevated': [],
            'decreased': [],
            'critical': []
        }
        
        all_markers = {
            **profile.tumor_markers,
            **profile.inflammatory_markers,
            **profile.metabolic_markers,
            **profile.protein_markers,
            **profile.hormone_levels,
            **profile.coagulation_markers
        }
        
        for marker, value in all_markers.items():
            if marker in self.reference_ranges:
                ref_range = self.reference_ranges[marker]
                
                if 'normal_max' in ref_range and value > ref_range['normal_max']:
                    severity = 'critical' if value > ref_range['normal_max'] * 3 else 'elevated'
                    abnormal[severity].append({
                        'marker': marker,
                        'value': value,
                        'reference_max': ref_range['normal_max'],
                        'fold_change': value / ref_range['normal_max'],
                        'unit': ref_range.get('unit', '')
                    })
                
                elif 'normal_min' in ref_range and value < ref_range['normal_min']:
                    severity = 'critical' if value < ref_range['normal_min'] * 0.5 else 'decreased'
                    abnormal[severity].append({
                        'marker': marker,
                        'value': value,
                        'reference_min': ref_range['normal_min'],
                        'fold_change': value / ref_range['normal_min'],
                        'unit': ref_range.get('unit', '')
                    })
        
        return abnormal

class ClinicalDecisionSupport:
    """Clinical decision support system"""
    
    def __init__(self):
        self.clinical_encoder = ClinicalDataEncoder()
        self.treatment_predictor = TreatmentResponsePredictor()
        self.biomarker_processor = BiomarkerProcessor()
        
        # Treatment guidelines
        self.treatment_guidelines = self._load_treatment_guidelines()
        
        # Drug interaction database
        self.drug_interactions = self._load_drug_interactions()
        
        # Contraindication rules
        self.contraindications = self._load_contraindications()
    
    def _load_treatment_guidelines(self) -> Dict[str, Dict]:
        """Load evidence-based treatment guidelines"""
        
        return {
            'breast_cancer': {
                'er_positive': {
                    'first_line': ['tamoxifen', 'aromatase_inhibitor'],
                    'second_line': ['fulvestrant', 'cdk4_6_inhibitor'],
                    'metastatic': ['chemotherapy', 'targeted_therapy']
                },
                'her2_positive': {
                    'first_line': ['trastuzumab', 'pertuzumab'],
                    'second_line': ['trastuzumab_emtansine'],
                    'resistance': ['tucatinib', 'capecitabine']
                },
                'triple_negative': {
                    'first_line': ['chemotherapy'],
                    'pd_l1_positive': ['immunotherapy'],
                    'brca_mutated': ['parp_inhibitor']
                }
            },
            'lung_cancer': {
                'nsclc': {
                    'egfr_mutated': ['osimertinib', 'erlotinib'],
                    'alk_rearranged': ['alectinib', 'crizotinib'],
                    'kras_g12c': ['sotorasib', 'adagrasib'],
                    'pd_l1_high': ['pembrolizumab', 'nivolumab']
                },
                'sclc': {
                    'extensive_stage': ['chemotherapy', 'immunotherapy'],
                    'limited_stage': ['chemoradiation']
                }
            },
            'colorectal_cancer': {
                'msi_high': ['immunotherapy'],
                'kras_wild_type': ['cetuximab', 'panitumumab'],
                'braf_mutated': ['encorafenib', 'cetuximab'],
                'metastatic': ['chemotherapy', 'bevacizumab']
            }
        }
    
    def _load_drug_interactions(self) -> Dict[str, List[str]]:
        """Load drug interaction database"""
        
        return {
            'warfarin': ['tamoxifen', 'fluorouracil', 'capecitabine'],
            'digoxin': ['verapamil', 'quinidine', 'amiodarone'],
            'phenytoin': ['fluorouracil', 'leucovorin'],
            'methotrexate': ['proton_pump_inhibitors', 'trimethoprim']
        }
    
    def _load_contraindications(self) -> Dict[str, List[str]]:
        """Load contraindication rules"""
        
        return {
            'doxorubicin': ['heart_failure', 'cardiomyopathy'],
            'bleomycin': ['pulmonary_fibrosis', 'lung_disease'],
            'cisplatin': ['kidney_disease', 'hearing_loss'],
            'methotrexate': ['liver_disease', 'kidney_disease'],
            'tamoxifen': ['thromboembolism_history', 'pregnancy']
        }
    
    def generate_treatment_recommendations(self, clinical_data: ClinicalData,
                                         biomarker_profile: BiomarkerProfile) -> Dict[str, Any]:
        """Generate evidence-based treatment recommendations"""
        
        recommendations = {
            'primary_recommendations': [],
            'alternative_options': [],
            'contraindications': [],
            'monitoring_requirements': [],
            'drug_interactions': [],
            'supportive_care': [],
            'clinical_trials': []
        }
        
        cancer_type = clinical_data.cancer_type.lower().replace(' ', '_')
        
        # Get guidelines for cancer type
        if cancer_type in self.treatment_guidelines:
            guidelines = self.treatment_guidelines[cancer_type]
            
            # Determine molecular subtype
            subtype = self._determine_molecular_subtype(clinical_data, biomarker_profile)
            
            if subtype in guidelines:
                subtype_guidelines = guidelines[subtype]
                
                # Primary recommendations
                if 'first_line' in subtype_guidelines:
                    recommendations['primary_recommendations'].extend(
                        subtype_guidelines['first_line']
                    )
                
                # Alternative options
                if 'second_line' in subtype_guidelines:
                    recommendations['alternative_options'].extend(
                        subtype_guidelines['second_line']
                    )
        
        # Check contraindications
        for treatment in recommendations['primary_recommendations']:
            if treatment in self.contraindications:
                contraindicated_conditions = self.contraindications[treatment]
                patient_conditions = [c.lower().replace(' ', '_') for c in clinical_data.comorbidities]
                
                for condition in contraindicated_conditions:
                    if condition in patient_conditions:
                        recommendations['contraindications'].append({
                            'treatment': treatment,
                            'condition': condition,
                            'severity': 'absolute'
                        })
        
        # Check drug interactions
        current_medications = [med['name'].lower() for med in clinical_data.medications]
        for treatment in recommendations['primary_recommendations']:
            if treatment in self.drug_interactions:
                interacting_drugs = self.drug_interactions[treatment]
                
                for drug in interacting_drugs:
                    if drug in current_medications:
                        recommendations['drug_interactions'].append({
                            'treatment': treatment,
                            'interacting_drug': drug,
                            'severity': 'moderate'
                        })
        
        # Monitoring requirements based on treatments
        recommendations['monitoring_requirements'] = self._generate_monitoring_requirements(
            recommendations['primary_recommendations'], clinical_data
        )
        
        # Supportive care recommendations
        recommendations['supportive_care'] = self._generate_supportive_care(
            clinical_data, biomarker_profile
        )
        
        return recommendations
    
    def _determine_molecular_subtype(self, clinical_data: ClinicalData,
                                   biomarker_profile: BiomarkerProfile) -> str:
        """Determine molecular subtype based on biomarkers"""
        
        cancer_type = clinical_data.cancer_type.lower()
        
        if 'breast' in cancer_type:
            # Breast cancer subtypes
            er_status = clinical_data.biomarkers.get('ER', {}).get('value', 'unknown')
            pr_status = clinical_data.biomarkers.get('PR', {}).get('value', 'unknown')
            her2_status = clinical_data.biomarkers.get('HER2', {}).get('value', 'unknown')
            
            if her2_status == 'positive':
                return 'her2_positive'
            elif er_status == 'positive' or pr_status == 'positive':
                return 'er_positive'
            else:
                return 'triple_negative'
        
        elif 'lung' in cancer_type:
            # Lung cancer subtypes
            egfr_status = clinical_data.biomarkers.get('EGFR', {}).get('value', 'unknown')
            alk_status = clinical_data.biomarkers.get('ALK', {}).get('value', 'unknown')
            kras_status = clinical_data.biomarkers.get('KRAS', {}).get('value', 'unknown')
            pdl1_status = clinical_data.biomarkers.get('PD-L1', {}).get('value', 0)
            
            if 'mutated' in egfr_status:
                return 'egfr_mutated'
            elif 'rearranged' in alk_status:
                return 'alk_rearranged'
            elif 'G12C' in kras_status:
                return 'kras_g12c'
            elif pdl1_status >= 50:
                return 'pd_l1_high'
            else:
                return 'nsclc'
        
        elif 'colorectal' in cancer_type:
            # Colorectal cancer subtypes
            msi_status = clinical_data.biomarkers.get('MSI', {}).get('value', 'stable')
            kras_status = clinical_data.biomarkers.get('KRAS', {}).get('value', 'unknown')
            braf_status = clinical_data.biomarkers.get('BRAF', {}).get('value', 'unknown')
            
            if msi_status == 'high':
                return 'msi_high'
            elif 'wild_type' in kras_status:
                return 'kras_wild_type'
            elif 'mutated' in braf_status:
                return 'braf_mutated'
            else:
                return 'metastatic'
        
        return 'standard'
    
    def _generate_monitoring_requirements(self, treatments: List[str],
                                        clinical_data: ClinicalData) -> List[Dict]:
        """Generate monitoring requirements for treatments"""
        
        monitoring = []
        
        monitoring_protocols = {
            'doxorubicin': {
                'test': 'echocardiogram',
                'frequency': 'every_3_months',
                'parameter': 'ejection_fraction',
                'threshold': 50
            },
            'cisplatin': {
                'test': 'audiometry',
                'frequency': 'before_each_cycle',
                'parameter': 'hearing_loss',
                'threshold': None
            },
            'methotrexate': {
                'test': 'liver_function',
                'frequency': 'weekly',
                'parameter': 'ALT',
                'threshold': 100
            },
            'tamoxifen': {
                'test': 'gynecologic_exam',
                'frequency': 'annually',
                'parameter': 'endometrial_thickness',
                'threshold': 5
            }
        }
        
        for treatment in treatments:
            if treatment in monitoring_protocols:
                monitoring.append(monitoring_protocols[treatment])
        
        # Add standard monitoring
        monitoring.append({
            'test': 'complete_blood_count',
            'frequency': 'before_each_cycle',
            'parameter': 'neutrophil_count',
            'threshold': 1500
        })
        
        monitoring.append({
            'test': 'comprehensive_metabolic_panel',
            'frequency': 'before_each_cycle',
            'parameter': 'creatinine',
            'threshold': 1.5
        })
        
        return monitoring
    
    def _generate_supportive_care(self, clinical_data: ClinicalData,
                                biomarker_profile: BiomarkerProfile) -> List[str]:
        """Generate supportive care recommendations"""
        
        supportive_care = []
        
        # Nutritional support
        if clinical_data.bmi < 18.5:
            supportive_care.append('Nutritional counseling and supplementation')
        
        # Anemia management
        if biomarker_profile.protein_markers.get('Hemoglobin', 15) < 10:
            supportive_care.append('Anemia evaluation and treatment')
        
        # Infection prevention
        if biomarker_profile.protein_markers.get('WBC', 5000) < 3000:
            supportive_care.append('Infection precautions and monitoring')
        
        # Pain management
        supportive_care.append('Pain assessment and management plan')
        
        # Psychological support
        supportive_care.append('Psychosocial support and counseling')
        
        # Fertility preservation
        if clinical_data.age < 45 and clinical_data.gender == 'female':
            supportive_care.append('Fertility preservation consultation')
        
        return supportive_care

def create_sample_clinical_data() -> ClinicalData:
    """Create sample clinical data for testing"""
    
    # Sample laboratory results
    lab_results = {
        'CBC': {
            'Hemoglobin': {'value': 11.5, 'date': datetime.now(), 'unit': 'g/dL', 'ref_range': '12-16'},
            'WBC': {'value': 4500, 'date': datetime.now(), 'unit': '/μL', 'ref_range': '4000-11000'},
            'Platelets': {'value': 180000, 'date': datetime.now(), 'unit': '/μL', 'ref_range': '150000-450000'}
        },
        'Chemistry': {
            'Creatinine': {'value': 1.0, 'date': datetime.now(), 'unit': 'mg/dL', 'ref_range': '0.6-1.2'},
            'ALT': {'value': 35, 'date': datetime.now(), 'unit': 'U/L', 'ref_range': '7-40'},
            'Albumin': {'value': 3.8, 'date': datetime.now(), 'unit': 'g/dL', 'ref_range': '3.5-5.0'}
        }
    }
    
    # Sample biomarkers
    biomarkers = {
        'ER': {'value': 'positive', 'method': 'IHC', 'date': datetime.now()},
        'PR': {'value': 'positive', 'method': 'IHC', 'date': datetime.now()},
        'HER2': {'value': 'negative', 'method': 'IHC', 'date': datetime.now()},
        'Ki-67': {'value': '15%', 'method': 'IHC', 'date': datetime.now()}
    }
    
    # Sample treatments
    treatments = [
        {
            'type': 'surgery',
            'name': 'lumpectomy',
            'start_date': datetime.now() - timedelta(days=30),
            'end_date': datetime.now() - timedelta(days=30),
            'outcome': 'complete_resection'
        },
        {
            'type': 'chemotherapy',
            'name': 'AC-T',
            'start_date': datetime.now() - timedelta(days=14),
            'cycles_planned': 8,
            'cycles_completed': 2
        }
    ]
    
    return ClinicalData(
        patient_id='CLIN_001',
        age=55.0,
        gender='female',
        race='caucasian',
        ethnicity='non_hispanic',
        cancer_type='breast_cancer',
        tumor_stage='T2N1M0',
        tumor_grade=2,
        histology='invasive_ductal_carcinoma',
        primary_site='left_breast_upper_outer',
        diagnosis_date=datetime.now() - timedelta(days=60),
        height_cm=165.0,
        weight_kg=68.0,
        bmi=25.0,
        performance_status=1,
        laboratory_results=lab_results,
        biomarkers=biomarkers,
        treatments=treatments,
        imaging_studies=[],
        pathology_reports=[],
        response_assessments=[],
        survival_status='alive',
        last_followup_date=datetime.now(),
        comorbidities=['hypertension', 'diabetes_type_2'],
        medications=[
            {'name': 'metformin', 'dose': '500mg', 'frequency': 'twice_daily'},
            {'name': 'lisinopril', 'dose': '10mg', 'frequency': 'daily'}
        ],
        allergies=['penicillin'],
        family_history=[
            {'relation': 'mother', 'condition': 'breast_cancer', 'age_at_diagnosis': 62}
        ],
        smoking_status='former',
        alcohol_use='occasional',
        drug_use='none',
        occupation='teacher'
    )

def run_clinical_integration_demo():
    """Run a demonstration of clinical integration"""
    
    print("=" * 60)
    print("CLINICAL INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample clinical data
    clinical_data = create_sample_clinical_data()
    
    # Initialize biomarker processor
    biomarker_processor = BiomarkerProcessor()
    
    # Sample biomarker values
    biomarker_values = {
        'CEA': 3.2,
        'CA19-9': 25.0,
        'CRP': 8.5,
        'LDH': 280.0,
        'Albumin': 3.8,
        'Hemoglobin': 11.5,
        'WBC': 4500,
        'Platelets': 180000,
        'ALT': 35,
        'Creatinine': 1.0
    }
    
    # Process biomarkers
    biomarker_profile = biomarker_processor.process_biomarkers(biomarker_values)
    
    # Calculate risk scores
    risk_scores = biomarker_processor.calculate_biomarker_risk_score(biomarker_profile)
    
    # Identify abnormal biomarkers
    abnormal_biomarkers = biomarker_processor.identify_abnormal_biomarkers(biomarker_profile)
    
    # Generate clinical recommendations
    decision_support = ClinicalDecisionSupport()
    recommendations = decision_support.generate_treatment_recommendations(
        clinical_data, biomarker_profile
    )
    
    # Display results
    print("PATIENT INFORMATION:")
    print(f"Patient ID: {clinical_data.patient_id}")
    print(f"Age: {clinical_data.age} years")
    print(f"Gender: {clinical_data.gender}")
    print(f"Cancer Type: {clinical_data.cancer_type}")
    print(f"Stage: {clinical_data.tumor_stage}")
    print(f"Performance Status: ECOG {clinical_data.performance_status}")
    print(f"BMI: {clinical_data.bmi}")
    print()
    
    print("BIOMARKER ANALYSIS:")
    print("Risk Scores:")
    for category, score in risk_scores.items():
        print(f"  {category.replace('_', ' ').title()}: {score:.3f}")
    print()
    
    print("Biomarker Categories:")
    print(f"  Tumor Markers: {len(biomarker_profile.tumor_markers)} values")
    print(f"  Inflammatory Markers: {len(biomarker_profile.inflammatory_markers)} values")
    print(f"  Metabolic Markers: {len(biomarker_profile.metabolic_markers)} values")
    print()
    
    print("ABNORMAL BIOMARKERS:")
    for severity, markers in abnormal_biomarkers.items():
        if markers:
            print(f"{severity.upper()}:")
            for marker_info in markers:
                print(f"  {marker_info['marker']}: {marker_info['value']} {marker_info['unit']} "
                      f"(Fold change: {marker_info['fold_change']:.2f})")
    print()
    
    print("TREATMENT RECOMMENDATIONS:")
    print("Primary Recommendations:")
    for rec in recommendations['primary_recommendations']:
        print(f"  • {rec.replace('_', ' ').title()}")
    
    print("\nAlternative Options:")
    for alt in recommendations['alternative_options']:
        print(f"  • {alt.replace('_', ' ').title()}")
    
    if recommendations['contraindications']:
        print("\nContraindications:")
        for contra in recommendations['contraindications']:
            print(f"  ⚠️ {contra['treatment']} - {contra['condition']}")
    
    if recommendations['drug_interactions']:
        print("\nDrug Interactions:")
        for interaction in recommendations['drug_interactions']:
            print(f"  ⚠️ {interaction['treatment']} with {interaction['interacting_drug']}")
    
    print("\nMonitoring Requirements:")
    for monitor in recommendations['monitoring_requirements']:
        print(f"  • {monitor['test']} - {monitor['frequency']}")
    
    print("\nSupportive Care:")
    for support in recommendations['supportive_care']:
        print(f"  • {support}")
    
    print()
    
    return {
        'clinical_data': clinical_data,
        'biomarker_profile': biomarker_profile,
        'risk_scores': risk_scores,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    # Run demonstration
    result = run_clinical_integration_demo()
