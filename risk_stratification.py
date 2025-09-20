"""
Risk Stratification Module
=========================

Comprehensive risk analysis and stratification for cancer patients including
survival prediction models, risk scoring algorithms, and prognostic factor analysis.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import CoxPHRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import concordance_index_censored, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import from config module
try:
    from config import *
except ImportError:
    # Fallback definitions if config module not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

@dataclass
class RiskAssessmentResult:
    """Results from comprehensive risk assessment"""
    patient_id: str
    assessment_date: datetime
    overall_risk_score: float
    risk_category: str  # Low, Moderate, High, Critical
    survival_probability_1yr: float
    survival_probability_3yr: float
    survival_probability_5yr: float
    progression_risk: float
    recurrence_risk: float
    metastasis_risk: float
    treatment_response_probability: float
    prognostic_factors: Dict[str, float]
    protective_factors: List[str]
    adverse_factors: List[str]
    biomarker_contributions: Dict[str, float]
    staging_contribution: float
    genomic_contribution: float
    clinical_contribution: float
    treatment_recommendations: List[str]
    monitoring_intensity: str
    confidence_interval: Tuple[float, float]
    uncertainty_score: float
    analysis_timestamp: datetime

@dataclass 
class PrognosticFactors:
    """Comprehensive prognostic factors"""
    # Clinical factors
    age: float
    gender: str
    performance_status: int  # ECOG 0-4
    comorbidity_score: float
    
    # Tumor characteristics
    tumor_size: float  # cm
    tumor_stage: str
    histologic_grade: int  # 1-4
    lymph_node_involvement: int
    metastases_present: bool
    tumor_location: str
    
    # Molecular markers
    ki67_index: float
    p53_status: str
    her2_status: str
    hormone_receptor_status: str
    microsatellite_instability: str
    tumor_mutational_burden: float
    
    # Treatment factors
    surgery_type: str
    chemotherapy_regimen: str
    radiation_dose: float
    targeted_therapy: List[str]
    immunotherapy: bool
    
    # Laboratory values
    hemoglobin: float
    white_cell_count: float
    platelet_count: float
    albumin: float
    ldh: float
    cea: float
    ca199: float
    
    # Lifestyle factors
    smoking_status: str
    alcohol_consumption: float
    bmi: float
    exercise_level: str

class SurvivalPredictionNetwork(nn.Module):
    """Deep learning model for survival prediction"""
    
    def __init__(self, input_dim: int = 128, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Feature encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_encoder = nn.Sequential(*layers)
        
        # Survival time predictor (log-normal distribution)
        self.survival_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # mu and log(sigma) for log-normal
        )
        
        # Event probability predictor (censoring)
        self.event_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Risk stratification classifier
        self.risk_classifier = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Low, Moderate, High, Critical
        )
        
        # Progression risk predictor
        self.progression_predictor = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multiple predictions"""
        
        # Encode features
        features = self.feature_encoder(x)
        
        # Survival prediction (log-normal parameters)
        survival_params = self.survival_predictor(features)
        mu = survival_params[:, 0]
        log_sigma = survival_params[:, 1]
        
        # Event probability
        event_prob = self.event_predictor(features).squeeze()
        
        # Risk classification
        risk_logits = self.risk_classifier(features)
        risk_probs = F.softmax(risk_logits, dim=1)
        
        # Progression risk
        progression_risk = self.progression_predictor(features).squeeze()
        
        return {
            'features': features,
            'survival_mu': mu,
            'survival_log_sigma': log_sigma,
            'event_probability': event_prob,
            'risk_logits': risk_logits,
            'risk_probabilities': risk_probs,
            'progression_risk': progression_risk
        }

class CoxProportionalHazardsModel:
    """Cox Proportional Hazards model for survival analysis"""
    
    def __init__(self):
        self.model = CoxPHRegressor()
        self.feature_scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
    def fit(self, features: np.ndarray, survival_times: np.ndarray, 
            events: np.ndarray, feature_names: List[str] = None):
        """Fit Cox model to survival data"""
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Fit Cox model
        self.model.fit(features_scaled, events, survival_times)
        
        self.is_fitted = True
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
        
        logger.info(f"Cox model fitted with {len(self.feature_names)} features")
        
    def predict_survival_function(self, features: np.ndarray, 
                                 time_points: np.ndarray) -> np.ndarray:
        """Predict survival probabilities at specific time points"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features_scaled = self.feature_scaler.transform(features)
        
        # Get survival function
        survival_functions = self.model.predict_survival_function(features_scaled)
        
        # Evaluate at specific time points
        survival_probs = np.zeros((len(features), len(time_points)))
        
        for i, sf in enumerate(survival_functions):
            survival_probs[i] = sf(time_points)
        
        return survival_probs
    
    def get_hazard_ratios(self) -> Dict[str, float]:
        """Get hazard ratios for each feature"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting hazard ratios")
        
        hazard_ratios = np.exp(self.model.coef_)
        
        return dict(zip(self.feature_names, hazard_ratios))

class PrognosticScoreCalculator:
    """Calculate various prognostic scores and indices"""
    
    @staticmethod
    def calculate_tnm_score(tumor_size: float, nodes_positive: int, 
                           metastases: bool, cancer_type: str) -> Tuple[str, float]:
        """Calculate TNM staging score"""
        
        # T stage
        if tumor_size <= 2.0:
            t_stage = "T1"
            t_score = 1
        elif tumor_size <= 5.0:
            t_stage = "T2"
            t_score = 2
        elif tumor_size <= 7.0:
            t_stage = "T3"
            t_score = 3
        else:
            t_stage = "T4"
            t_score = 4
        
        # N stage
        if nodes_positive == 0:
            n_stage = "N0"
            n_score = 0
        elif nodes_positive <= 3:
            n_stage = "N1"
            n_score = 1
        elif nodes_positive <= 9:
            n_stage = "N2"
            n_score = 2
        else:
            n_stage = "N3"
            n_score = 3
        
        # M stage
        m_stage = "M1" if metastases else "M0"
        m_score = 1 if metastases else 0
        
        # Calculate overall stage
        total_score = t_score + n_score + m_score
        
        if total_score <= 1:
            stage = "Stage I"
        elif total_score <= 3:
            stage = "Stage II"
        elif total_score <= 5:
            stage = "Stage III"
        else:
            stage = "Stage IV"
        
        tnm_string = f"{t_stage}{n_stage}{m_stage}"
        
        return f"{stage} ({tnm_string})", total_score / 8.0  # Normalized score
    
    @staticmethod
    def calculate_nottingham_score(tubule_formation: int, nuclear_grade: int,
                                  mitotic_rate: int) -> Tuple[int, str]:
        """Calculate Nottingham Prognostic Index for breast cancer"""
        
        # Validate inputs (1-3 scale)
        tubule_formation = max(1, min(3, tubule_formation))
        nuclear_grade = max(1, min(3, nuclear_grade))
        mitotic_rate = max(1, min(3, mitotic_rate))
        
        total_score = tubule_formation + nuclear_grade + mitotic_rate
        
        if total_score <= 5:
            grade = "Grade 1 (Well differentiated)"
        elif total_score <= 7:
            grade = "Grade 2 (Moderately differentiated)"
        else:
            grade = "Grade 3 (Poorly differentiated)"
        
        return total_score, grade
    
    @staticmethod
    def calculate_karnofsky_score(performance_data: Dict) -> int:
        """Calculate Karnofsky Performance Status"""
        
        # This is a simplified version - real implementation would use detailed assessment
        activity_level = performance_data.get('activity_level', 'normal')
        care_needed = performance_data.get('care_needed', False)
        symptoms = performance_data.get('symptoms', 'none')
        
        if activity_level == 'normal' and not care_needed:
            return 100  # Normal activity
        elif activity_level == 'normal' and symptoms == 'minor':
            return 90   # Normal activity with minor symptoms
        elif activity_level == 'reduced' and not care_needed:
            return 80   # Normal activity with effort
        elif activity_level == 'reduced' and symptoms == 'significant':
            return 70   # Cares for self, unable to work
        elif care_needed and activity_level == 'limited':
            return 60   # Requires occasional assistance
        else:
            return 50   # Requires considerable assistance
    
    @staticmethod
    def calculate_charlson_comorbidity_index(comorbidities: List[str]) -> int:
        """Calculate Charlson Comorbidity Index"""
        
        # Comorbidity weights
        weights = {
            'myocardial_infarction': 1,
            'congestive_heart_failure': 1,
            'peripheral_vascular_disease': 1,
            'cerebrovascular_disease': 1,
            'dementia': 1,
            'chronic_pulmonary_disease': 1,
            'rheumatic_disease': 1,
            'peptic_ulcer_disease': 1,
            'mild_liver_disease': 1,
            'diabetes': 1,
            'diabetes_complications': 2,
            'hemiplegia': 2,
            'renal_disease': 2,
            'malignancy': 2,
            'moderate_liver_disease': 3,
            'metastatic_solid_tumor': 6,
            'aids': 6
        }
        
        total_score = sum(weights.get(condition, 0) for condition in comorbidities)
        return total_score

class RiskStratificationPipeline:
    """Comprehensive risk stratification pipeline"""
    
    def __init__(self):
        self.survival_model = SurvivalPredictionNetwork()
        self.cox_model = CoxProportionalHazardsModel()
        self.score_calculator = PrognosticScoreCalculator()
        self.feature_importance = {}
        self.risk_thresholds = {
            'low': 0.25,
            'moderate': 0.50,
            'high': 0.75
        }
        
    def extract_features(self, factors: PrognosticFactors) -> np.ndarray:
        """Extract numerical features from prognostic factors"""
        
        features = []
        
        # Clinical features
        features.extend([
            factors.age / 100.0,  # Normalize age
            1.0 if factors.gender == 'male' else 0.0,
            factors.performance_status / 4.0,  # ECOG scale
            factors.comorbidity_score / 10.0,  # Normalize
        ])
        
        # Tumor features
        features.extend([
            factors.tumor_size / 10.0,  # Normalize size
            float(factors.histologic_grade) / 4.0,
            float(factors.lymph_node_involvement) / 20.0,  # Max ~20 nodes
            1.0 if factors.metastases_present else 0.0,
        ])
        
        # Molecular features
        features.extend([
            factors.ki67_index / 100.0,
            1.0 if factors.p53_status == 'mutated' else 0.0,
            1.0 if factors.her2_status == 'positive' else 0.0,
            1.0 if factors.hormone_receptor_status == 'positive' else 0.0,
            1.0 if factors.microsatellite_instability == 'high' else 0.0,
            min(factors.tumor_mutational_burden / 100.0, 1.0),  # Cap at 100
        ])
        
        # Laboratory features (normalized)
        features.extend([
            factors.hemoglobin / 20.0,  # g/dL
            factors.white_cell_count / 20000.0,  # per μL
            factors.platelet_count / 500000.0,  # per μL
            factors.albumin / 5.0,  # g/dL
            factors.ldh / 1000.0,  # U/L
            factors.cea / 100.0,  # ng/mL
            factors.ca199 / 1000.0,  # U/mL
        ])
        
        # Lifestyle features
        features.extend([
            1.0 if factors.smoking_status == 'current' else 0.5 if factors.smoking_status == 'former' else 0.0,
            min(factors.alcohol_consumption / 7.0, 1.0),  # drinks per week
            factors.bmi / 50.0,  # kg/m²
            {'sedentary': 0.0, 'light': 0.33, 'moderate': 0.67, 'vigorous': 1.0}.get(factors.exercise_level, 0.0)
        ])
        
        # Treatment features
        features.extend([
            1.0 if factors.surgery_type in ['radical', 'extensive'] else 0.5 if factors.surgery_type else 0.0,
            1.0 if factors.chemotherapy_regimen else 0.0,
            factors.radiation_dose / 70.0,  # Gy
            len(factors.targeted_therapy) / 5.0,  # Normalize by max ~5 drugs
            1.0 if factors.immunotherapy else 0.0,
        ])
        
        return np.array(features, dtype=np.float32)
    
    def assess_risk(self, factors: PrognosticFactors) -> RiskAssessmentResult:
        """Perform comprehensive risk assessment"""
        
        # Extract features
        features = self.extract_features(factors)
        
        # Calculate traditional prognostic scores
        tnm_stage, tnm_score = self.score_calculator.calculate_tnm_score(
            factors.tumor_size, factors.lymph_node_involvement, 
            factors.metastases_present, "lung"  # Example cancer type
        )
        
        karnofsky_score = self.score_calculator.calculate_karnofsky_score({
            'activity_level': 'normal' if factors.performance_status <= 1 else 'reduced',
            'care_needed': factors.performance_status >= 3,
            'symptoms': 'minor' if factors.performance_status <= 1 else 'significant'
        })
        
        # Calculate molecular risk scores
        molecular_risk = self._calculate_molecular_risk(factors)
        
        # Calculate clinical risk scores
        clinical_risk = self._calculate_clinical_risk(factors, tnm_score, karnofsky_score)
        
        # Calculate genomic risk scores
        genomic_risk = self._calculate_genomic_risk(factors)
        
        # Combine risk scores
        overall_risk = self._combine_risk_scores(clinical_risk, molecular_risk, genomic_risk)
        
        # Determine risk category
        risk_category = self._determine_risk_category(overall_risk)
        
        # Predict survival probabilities
        survival_1yr, survival_3yr, survival_5yr = self._predict_survival_probabilities(
            features, overall_risk
        )
        
        # Calculate specific risks
        progression_risk = self._calculate_progression_risk(factors, overall_risk)
        recurrence_risk = self._calculate_recurrence_risk(factors, overall_risk)
        metastasis_risk = self._calculate_metastasis_risk(factors, overall_risk)
        
        # Predict treatment response
        treatment_response_prob = self._predict_treatment_response(factors)
        
        # Identify prognostic factors
        prognostic_factors_dict = self._identify_prognostic_factors(factors, features)
        
        # Generate recommendations
        treatment_recommendations = self._generate_treatment_recommendations(
            factors, risk_category, overall_risk
        )
        
        # Determine monitoring intensity
        monitoring_intensity = self._determine_monitoring_intensity(risk_category, overall_risk)
        
        # Calculate confidence interval and uncertainty
        confidence_interval, uncertainty_score = self._calculate_uncertainty(
            features, overall_risk
        )
        
        return RiskAssessmentResult(
            patient_id=f"PATIENT_{hash(str(factors)) % 10000:04d}",
            assessment_date=datetime.now(),
            overall_risk_score=overall_risk,
            risk_category=risk_category,
            survival_probability_1yr=survival_1yr,
            survival_probability_3yr=survival_3yr,
            survival_probability_5yr=survival_5yr,
            progression_risk=progression_risk,
            recurrence_risk=recurrence_risk,
            metastasis_risk=metastasis_risk,
            treatment_response_probability=treatment_response_prob,
            prognostic_factors=prognostic_factors_dict,
            protective_factors=self._identify_protective_factors(factors),
            adverse_factors=self._identify_adverse_factors(factors),
            biomarker_contributions={
                'molecular': molecular_risk,
                'clinical': clinical_risk,
                'genomic': genomic_risk
            },
            staging_contribution=tnm_score,
            genomic_contribution=genomic_risk,
            clinical_contribution=clinical_risk,
            treatment_recommendations=treatment_recommendations,
            monitoring_intensity=monitoring_intensity,
            confidence_interval=confidence_interval,
            uncertainty_score=uncertainty_score,
            analysis_timestamp=datetime.now()
        )
    
    def _calculate_molecular_risk(self, factors: PrognosticFactors) -> float:
        """Calculate molecular risk score"""
        
        risk_score = 0.0
        
        # Ki-67 proliferation index
        risk_score += factors.ki67_index / 100.0 * 0.3
        
        # p53 mutation status
        if factors.p53_status == 'mutated':
            risk_score += 0.2
        
        # HER2 status (protective for some treatments)
        if factors.her2_status == 'positive':
            risk_score += 0.1  # Increased risk but targetable
        
        # Hormone receptor status (generally protective)
        if factors.hormone_receptor_status == 'negative':
            risk_score += 0.15
        
        # Microsatellite instability (complex - can be protective with immunotherapy)
        if factors.microsatellite_instability == 'high':
            risk_score += 0.05  # Mixed prognostic value
        
        # Tumor mutational burden
        if factors.tumor_mutational_burden > 20:
            risk_score += 0.1
        elif factors.tumor_mutational_burden > 10:
            risk_score += 0.05
        
        return min(risk_score, 1.0)
    
    def _calculate_clinical_risk(self, factors: PrognosticFactors, 
                                tnm_score: float, karnofsky_score: int) -> float:
        """Calculate clinical risk score"""
        
        risk_score = 0.0
        
        # Age factor
        if factors.age > 70:
            risk_score += 0.2
        elif factors.age > 60:
            risk_score += 0.1
        
        # Performance status
        risk_score += factors.performance_status / 4.0 * 0.3
        
        # TNM staging
        risk_score += tnm_score * 0.4
        
        # Comorbidity burden
        risk_score += min(factors.comorbidity_score / 10.0, 0.2)
        
        # Laboratory values indicating poor prognosis
        if factors.hemoglobin < 10.0:  # Anemia
            risk_score += 0.1
        
        if factors.albumin < 3.5:  # Hypoalbuminemia
            risk_score += 0.1
        
        if factors.ldh > 300:  # Elevated LDH
            risk_score += 0.1
        
        # Tumor markers
        if factors.cea > 5.0:
            risk_score += 0.05
        
        if factors.ca199 > 37.0:
            risk_score += 0.05
        
        return min(risk_score, 1.0)
    
    def _calculate_genomic_risk(self, factors: PrognosticFactors) -> float:
        """Calculate genomic risk score based on molecular features"""
        
        # This is a simplified version - real implementation would use 
        # comprehensive genomic profiling data
        risk_score = 0.0
        
        # Tumor mutational burden
        tmb_risk = min(factors.tumor_mutational_burden / 50.0, 0.3)
        risk_score += tmb_risk
        
        # Microsatellite instability
        if factors.microsatellite_instability == 'high':
            risk_score += 0.1
        
        # p53 pathway disruption
        if factors.p53_status == 'mutated':
            risk_score += 0.2
        
        # Proliferation signature (approximated by Ki-67)
        proliferation_risk = factors.ki67_index / 100.0 * 0.2
        risk_score += proliferation_risk
        
        return min(risk_score, 1.0)
    
    def _combine_risk_scores(self, clinical: float, molecular: float, genomic: float) -> float:
        """Combine different risk scores into overall risk"""
        
        # Weighted combination
        weights = {
            'clinical': 0.5,
            'molecular': 0.3,
            'genomic': 0.2
        }
        
        overall_risk = (
            clinical * weights['clinical'] +
            molecular * weights['molecular'] +
            genomic * weights['genomic']
        )
        
        return min(overall_risk, 1.0)
    
    def _determine_risk_category(self, overall_risk: float) -> str:
        """Determine risk category based on overall risk score"""
        
        if overall_risk < self.risk_thresholds['low']:
            return "Low"
        elif overall_risk < self.risk_thresholds['moderate']:
            return "Moderate"
        elif overall_risk < self.risk_thresholds['high']:
            return "High"
        else:
            return "Critical"
    
    def _predict_survival_probabilities(self, features: np.ndarray, 
                                       overall_risk: float) -> Tuple[float, float, float]:
        """Predict survival probabilities at 1, 3, and 5 years"""
        
        # Simplified survival prediction based on risk score
        # Real implementation would use trained survival models
        
        base_survival = 1.0 - overall_risk
        
        # Time-dependent survival (exponential decay)
        survival_1yr = base_survival ** 0.5
        survival_3yr = base_survival ** 1.5
        survival_5yr = base_survival ** 2.5
        
        return survival_1yr, survival_3yr, survival_5yr
    
    def _calculate_progression_risk(self, factors: PrognosticFactors, overall_risk: float) -> float:
        """Calculate risk of disease progression"""
        
        progression_risk = overall_risk * 0.8  # Slightly lower than overall risk
        
        # Adjust based on specific factors
        if factors.metastases_present:
            progression_risk += 0.2
        
        if factors.tumor_size > 5.0:
            progression_risk += 0.1
        
        if factors.lymph_node_involvement > 3:
            progression_risk += 0.15
        
        return min(progression_risk, 1.0)
    
    def _calculate_recurrence_risk(self, factors: PrognosticFactors, overall_risk: float) -> float:
        """Calculate risk of disease recurrence"""
        
        recurrence_risk = overall_risk * 0.7
        
        # Adjust based on treatment factors
        if not factors.chemotherapy_regimen:
            recurrence_risk += 0.2
        
        if factors.radiation_dose < 50:  # Insufficient radiation
            recurrence_risk += 0.15
        
        # Molecular factors
        if factors.hormone_receptor_status == 'positive' and factors.her2_status == 'negative':
            recurrence_risk -= 0.1  # Better prognosis
        
        return max(0.0, min(recurrence_risk, 1.0))
    
    def _calculate_metastasis_risk(self, factors: PrognosticFactors, overall_risk: float) -> float:
        """Calculate risk of metastatic spread"""
        
        if factors.metastases_present:
            return 1.0  # Already metastatic
        
        metastasis_risk = overall_risk * 0.6
        
        # High-risk features for metastasis
        if factors.tumor_size > 7.0:
            metastasis_risk += 0.2
        
        if factors.lymph_node_involvement > 5:
            metastasis_risk += 0.25
        
        if factors.histologic_grade >= 3:
            metastasis_risk += 0.15
        
        # Molecular features associated with metastasis
        if factors.ki67_index > 50:
            metastasis_risk += 0.1
        
        return min(metastasis_risk, 1.0)
    
    def _predict_treatment_response(self, factors: PrognosticFactors) -> float:
        """Predict probability of treatment response"""
        
        response_prob = 0.5  # Base probability
        
        # Performance status
        if factors.performance_status <= 1:
            response_prob += 0.2
        elif factors.performance_status >= 3:
            response_prob -= 0.3
        
        # Molecular predictors
        if factors.her2_status == 'positive' and factors.targeted_therapy:
            response_prob += 0.3
        
        if factors.hormone_receptor_status == 'positive':
            response_prob += 0.2
        
        if factors.microsatellite_instability == 'high' and factors.immunotherapy:
            response_prob += 0.4
        
        # Tumor burden
        if factors.metastases_present:
            response_prob -= 0.2
        
        if factors.tumor_size > 5.0:
            response_prob -= 0.1
        
        # Age and comorbidities
        if factors.age > 70:
            response_prob -= 0.1
        
        if factors.comorbidity_score > 5:
            response_prob -= 0.15
        
        return max(0.1, min(response_prob, 0.9))
    
    def _identify_prognostic_factors(self, factors: PrognosticFactors, 
                                    features: np.ndarray) -> Dict[str, float]:
        """Identify key prognostic factors and their contributions"""
        
        # This would normally use feature importance from trained models
        prognostic_factors = {}
        
        # Clinical factors
        prognostic_factors['age'] = factors.age / 100.0
        prognostic_factors['performance_status'] = factors.performance_status / 4.0
        prognostic_factors['tumor_size'] = min(factors.tumor_size / 10.0, 1.0)
        prognostic_factors['lymph_nodes'] = min(factors.lymph_node_involvement / 20.0, 1.0)
        prognostic_factors['histologic_grade'] = factors.histologic_grade / 4.0
        prognostic_factors['metastases'] = 1.0 if factors.metastases_present else 0.0
        
        # Molecular factors
        prognostic_factors['ki67'] = factors.ki67_index / 100.0
        prognostic_factors['p53_mutation'] = 1.0 if factors.p53_status == 'mutated' else 0.0
        prognostic_factors['her2_status'] = 1.0 if factors.her2_status == 'positive' else 0.0
        prognostic_factors['hormone_receptors'] = 1.0 if factors.hormone_receptor_status == 'positive' else 0.0
        
        # Laboratory factors
        prognostic_factors['hemoglobin'] = 1.0 - (factors.hemoglobin / 20.0)  # Lower is worse
        prognostic_factors['albumin'] = 1.0 - (factors.albumin / 5.0)  # Lower is worse
        prognostic_factors['ldh'] = min(factors.ldh / 1000.0, 1.0)  # Higher is worse
        
        return prognostic_factors
    
    def _identify_protective_factors(self, factors: PrognosticFactors) -> List[str]:
        """Identify protective prognostic factors"""
        
        protective = []
        
        if factors.age < 50:
            protective.append("Young age")
        
        if factors.performance_status <= 1:
            protective.append("Excellent performance status")
        
        if factors.tumor_size < 2.0:
            protective.append("Small tumor size")
        
        if factors.lymph_node_involvement == 0:
            protective.append("No lymph node involvement")
        
        if factors.histologic_grade <= 2:
            protective.append("Well to moderately differentiated")
        
        if not factors.metastases_present:
            protective.append("No distant metastases")
        
        if factors.hormone_receptor_status == 'positive':
            protective.append("Hormone receptor positive")
        
        if factors.her2_status == 'positive':
            protective.append("HER2 positive (targetable)")
        
        if factors.microsatellite_instability == 'high':
            protective.append("High MSI (immunotherapy responsive)")
        
        if factors.hemoglobin >= 12.0:
            protective.append("Normal hemoglobin")
        
        if factors.albumin >= 3.5:
            protective.append("Normal albumin")
        
        if factors.smoking_status == 'never':
            protective.append("Never smoker")
        
        if factors.bmi >= 18.5 and factors.bmi <= 25.0:
            protective.append("Normal BMI")
        
        return protective
    
    def _identify_adverse_factors(self, factors: PrognosticFactors) -> List[str]:
        """Identify adverse prognostic factors"""
        
        adverse = []
        
        if factors.age > 70:
            adverse.append("Advanced age")
        
        if factors.performance_status >= 2:
            adverse.append("Poor performance status")
        
        if factors.tumor_size > 5.0:
            adverse.append("Large tumor size")
        
        if factors.lymph_node_involvement > 3:
            adverse.append("Extensive lymph node involvement")
        
        if factors.histologic_grade >= 3:
            adverse.append("Poorly differentiated tumor")
        
        if factors.metastases_present:
            adverse.append("Distant metastases present")
        
        if factors.p53_status == 'mutated':
            adverse.append("p53 mutation")
        
        if factors.ki67_index > 50:
            adverse.append("High proliferation index")
        
        if factors.hormone_receptor_status == 'negative' and factors.her2_status == 'negative':
            adverse.append("Triple negative subtype")
        
        if factors.tumor_mutational_burden > 20:
            adverse.append("High tumor mutational burden")
        
        if factors.hemoglobin < 10.0:
            adverse.append("Anemia")
        
        if factors.albumin < 3.0:
            adverse.append("Hypoalbuminemia")
        
        if factors.ldh > 400:
            adverse.append("Elevated LDH")
        
        if factors.comorbidity_score > 5:
            adverse.append("Multiple comorbidities")
        
        if factors.smoking_status == 'current':
            adverse.append("Current smoker")
        
        return adverse
    
    def _generate_treatment_recommendations(self, factors: PrognosticFactors,
                                          risk_category: str, overall_risk: float) -> List[str]:
        """Generate personalized treatment recommendations"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_category == "Low":
            recommendations.extend([
                "Consider active surveillance with regular monitoring",
                "Standard therapy with de-escalation if appropriate",
                "Focus on quality of life preservation"
            ])
        elif risk_category == "Moderate":
            recommendations.extend([
                "Standard multimodal therapy recommended",
                "Consider clinical trial participation",
                "Regular monitoring for progression"
            ])
        elif risk_category == "High":
            recommendations.extend([
                "Aggressive multimodal therapy indicated",
                "Consider intensified treatment regimens",
                "Frequent monitoring and early intervention",
                "Strong recommendation for clinical trial"
            ])
        else:  # Critical
            recommendations.extend([
                "Urgent aggressive therapy required",
                "Consider experimental therapies",
                "Palliative care consultation",
                "Intensive monitoring and support"
            ])
        
        # Molecular-based recommendations
        if factors.her2_status == 'positive':
            recommendations.append("Anti-HER2 targeted therapy strongly recommended")
        
        if factors.hormone_receptor_status == 'positive':
            recommendations.append("Endocrine therapy indicated")
        
        if factors.microsatellite_instability == 'high':
            recommendations.append("Immunotherapy (PD-1/PD-L1 inhibitors) recommended")
        
        if factors.tumor_mutational_burden > 10:
            recommendations.append("Consider immunotherapy based on high TMB")
        
        # Performance status considerations
        if factors.performance_status >= 2:
            recommendations.append("Consider reduced-intensity regimens due to performance status")
        
        # Age-based considerations
        if factors.age > 75:
            recommendations.append("Geriatric oncology consultation recommended")
        
        return recommendations
    
    def _determine_monitoring_intensity(self, risk_category: str, overall_risk: float) -> str:
        """Determine appropriate monitoring intensity"""
        
        if risk_category == "Low":
            return "Standard monitoring (every 3-6 months)"
        elif risk_category == "Moderate":
            return "Enhanced monitoring (every 2-3 months)"
        elif risk_category == "High":
            return "Intensive monitoring (monthly)"
        else:  # Critical
            return "Very intensive monitoring (bi-weekly to weekly)"
    
    def _calculate_uncertainty(self, features: np.ndarray, 
                              overall_risk: float) -> Tuple[Tuple[float, float], float]:
        """Calculate prediction uncertainty and confidence intervals"""
        
        # Simplified uncertainty calculation
        # Real implementation would use model uncertainty estimation
        
        # Base uncertainty on risk score - higher risk has higher uncertainty
        base_uncertainty = 0.1 + (overall_risk * 0.2)
        
        # Confidence interval (95%)
        margin = 1.96 * base_uncertainty
        ci_lower = max(0.0, overall_risk - margin)
        ci_upper = min(1.0, overall_risk + margin)
        
        # Uncertainty score (0-1, where 1 is most uncertain)
        uncertainty_score = base_uncertainty
        
        return (ci_lower, ci_upper), uncertainty_score

def create_sample_prognostic_factors() -> PrognosticFactors:
    """Create sample prognostic factors for testing"""
    
    return PrognosticFactors(
        # Clinical factors
        age=65.0,
        gender='female',
        performance_status=1,
        comorbidity_score=3.0,
        
        # Tumor characteristics
        tumor_size=3.2,
        tumor_stage='T2N1M0',
        histologic_grade=2,
        lymph_node_involvement=2,
        metastases_present=False,
        tumor_location='upper_outer_quadrant',
        
        # Molecular markers
        ki67_index=25.0,
        p53_status='wild_type',
        her2_status='positive',
        hormone_receptor_status='positive',
        microsatellite_instability='stable',
        tumor_mutational_burden=8.5,
        
        # Treatment factors
        surgery_type='radical_mastectomy',
        chemotherapy_regimen='AC-T',
        radiation_dose=50.0,
        targeted_therapy=['trastuzumab', 'pertuzumab'],
        immunotherapy=False,
        
        # Laboratory values
        hemoglobin=12.5,
        white_cell_count=6500.0,
        platelet_count=250000.0,
        albumin=4.0,
        ldh=220.0,
        cea=2.1,
        ca199=15.0,
        
        # Lifestyle factors
        smoking_status='former',
        alcohol_consumption=2.0,
        bmi=24.5,
        exercise_level='moderate'
    )

def run_risk_stratification_demo():
    """Run a demonstration of risk stratification analysis"""
    
    print("=" * 60)
    print("RISK STRATIFICATION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create pipeline
    risk_pipeline = RiskStratificationPipeline()
    
    # Create sample prognostic factors
    factors = create_sample_prognostic_factors()
    
    print("PATIENT CHARACTERISTICS:")
    print(f"Age: {factors.age} years")
    print(f"Gender: {factors.gender}")
    print(f"Performance Status: ECOG {factors.performance_status}")
    print(f"Tumor Size: {factors.tumor_size} cm")
    print(f"Stage: {factors.tumor_stage}")
    print(f"Histologic Grade: {factors.histologic_grade}")
    print(f"Lymph Nodes: {factors.lymph_node_involvement} positive")
    print(f"Metastases: {'Yes' if factors.metastases_present else 'No'}")
    print(f"Ki-67: {factors.ki67_index}%")
    print(f"HER2: {factors.her2_status}")
    print(f"Hormone Receptors: {factors.hormone_receptor_status}")
    print()
    
    # Run risk assessment
    result = risk_pipeline.assess_risk(factors)
    
    # Display results
    print("RISK ASSESSMENT RESULTS:")
    print(f"Overall Risk Score: {result.overall_risk_score:.3f}")
    print(f"Risk Category: {result.risk_category}")
    print(f"1-Year Survival: {result.survival_probability_1yr:.1%}")
    print(f"3-Year Survival: {result.survival_probability_3yr:.1%}")
    print(f"5-Year Survival: {result.survival_probability_5yr:.1%}")
    print(f"Progression Risk: {result.progression_risk:.1%}")
    print(f"Recurrence Risk: {result.recurrence_risk:.1%}")
    print(f"Metastasis Risk: {result.metastasis_risk:.1%}")
    print(f"Treatment Response Probability: {result.treatment_response_probability:.1%}")
    print()
    
    print("RISK SCORE COMPONENTS:")
    for component, score in result.biomarker_contributions.items():
        print(f"{component.title()}: {score:.3f}")
    print(f"Staging Contribution: {result.staging_contribution:.3f}")
    print()
    
    print("KEY PROGNOSTIC FACTORS:")
    for factor, value in sorted(result.prognostic_factors.items(), 
                               key=lambda x: x[1], reverse=True)[:5]:
        print(f"{factor.replace('_', ' ').title()}: {value:.3f}")
    print()
    
    print("PROTECTIVE FACTORS:")
    for i, factor in enumerate(result.protective_factors[:5], 1):
        print(f"{i}. {factor}")
    print()
    
    print("ADVERSE FACTORS:")
    for i, factor in enumerate(result.adverse_factors[:5], 1):
        print(f"{i}. {factor}")
    print()
    
    print("TREATMENT RECOMMENDATIONS:")
    for i, recommendation in enumerate(result.treatment_recommendations, 1):
        print(f"{i}. {recommendation}")
    print()
    
    print(f"MONITORING INTENSITY: {result.monitoring_intensity}")
    print(f"UNCERTAINTY SCORE: {result.uncertainty_score:.3f}")
    print(f"CONFIDENCE INTERVAL: ({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})")
    print()
    
    return result

if __name__ == "__main__":
    # Run demonstration
    result = run_risk_stratification_demo()
