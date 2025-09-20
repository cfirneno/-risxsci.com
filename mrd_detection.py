"""
MRD Detection Module
===================

Minimal Residual Disease detection for cancer monitoring including
circulating tumor DNA analysis, liquid biopsy processing, and
longitudinal disease monitoring.

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

# Import from config module
try:
    from config import *
except ImportError:
    # Fallback definitions if config module not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger(__name__)

@dataclass
class MRDAnalysisResult:
    """Results from MRD analysis"""
    sample_id: str
    collection_date: datetime
    ctdna_detected: bool
    ctdna_concentration: float  # copies/mL
    variant_allele_frequency: float
    tumor_fraction: float
    mrd_status: str  # Positive, Negative, Indeterminate
    detection_sensitivity: float
    tracking_mutations: List[Dict]
    clonal_evolution: Dict[str, float]
    treatment_response: str
    relapse_risk: float
    monitoring_recommendations: List[str]
    longitudinal_trend: str
    confidence_score: float
    analysis_timestamp: datetime
    technical_quality: Dict[str, float]

@dataclass
class LiquidBiopsyData:
    """Liquid biopsy sample data"""
    sample_id: str
    patient_id: str
    collection_date: datetime
    sample_type: str  # plasma, serum, urine, CSF
    volume_ml: float
    processing_delay_hours: float
    storage_conditions: str
    cfDNA_concentration: float  # ng/mL
    cfDNA_integrity: float
    white_cell_contamination: float
    mutations_detected: List[Dict]
    copy_number_variants: List[Dict]
    methylation_markers: List[Dict]
    protein_biomarkers: Dict[str, float]

class CirculatingTumorDNAAnalyzer(nn.Module):
    """Advanced ctDNA analysis using deep learning"""
    
    def __init__(self, sequence_length: int = 150, hidden_dim: int = 256):
        super().__init__()
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # DNA sequence encoder
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Variant frequency predictor
        self.vaf_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Tumor fraction estimator
        self.tumor_fraction_estimator = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Clonal evolution tracker
        self.clonal_tracker = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # MRD classifier
        self.mrd_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # Positive, Negative, Indeterminate
        )
    
    def forward(self, sequence_data: torch.Tensor, 
                clinical_features: torch.Tensor,
                temporal_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Encode DNA sequences
        seq_features = self.sequence_encoder(sequence_data)
        seq_features = seq_features.squeeze(-1)
        
        # Predict variant allele frequency
        vaf = self.vaf_predictor(seq_features)
        
        # Estimate tumor fraction
        tumor_fraction = self.tumor_fraction_estimator(seq_features)
        
        results = {
            'sequence_features': seq_features,
            'variant_allele_frequency': vaf,
            'tumor_fraction': tumor_fraction
        }
        
        # Process temporal data if available
        if temporal_data is not None:
            lstm_out, _ = self.clonal_tracker(temporal_data)
            clonal_features = lstm_out[:, -1, :]  # Last time step
            
            # Combine features for MRD classification
            combined_features = torch.cat([seq_features, clonal_features], dim=1)
            mrd_logits = self.mrd_classifier(combined_features)
            
            results.update({
                'clonal_features': clonal_features,
                'mrd_logits': mrd_logits,
                'mrd_probabilities': F.softmax(mrd_logits, dim=1)
            })
        
        return results

class MRDDetectionPipeline:
    """Comprehensive MRD detection and monitoring pipeline"""
    
    def __init__(self):
        self.ctdna_analyzer = CirculatingTumorDNAAnalyzer()
        self.detection_thresholds = {
            'high_sensitivity': 1e-6,
            'standard': 1e-4,
            'clinical': 1e-3
        }
        self.mutation_tracker = defaultdict(list)
        self.longitudinal_data = defaultdict(list)
        
    def analyze_liquid_biopsy(self, biopsy_data: LiquidBiopsyData) -> MRDAnalysisResult:
        """Analyze liquid biopsy sample for MRD detection"""
        
        # Quality control checks
        qc_results = self._perform_quality_control(biopsy_data)
        
        if not qc_results['passed']:
            return self._create_failed_result(biopsy_data, qc_results)
        
        # Detect circulating tumor DNA
        ctdna_results = self._detect_ctdna(biopsy_data)
        
        # Analyze mutation patterns
        mutation_analysis = self._analyze_mutations(biopsy_data.mutations_detected)
        
        # Assess clonal evolution
        clonal_evolution = self._assess_clonal_evolution(
            biopsy_data.patient_id, 
            biopsy_data.mutations_detected
        )
        
        # Determine MRD status
        mrd_status = self._determine_mrd_status(ctdna_results, mutation_analysis)
        
        # Calculate relapse risk
        relapse_risk = self._calculate_relapse_risk(
            ctdna_results, mutation_analysis, clonal_evolution
        )
        
        # Generate monitoring recommendations
        recommendations = self._generate_monitoring_recommendations(
            mrd_status, relapse_risk, biopsy_data
        )
        
        # Assess longitudinal trends
        longitudinal_trend = self._assess_longitudinal_trends(
            biopsy_data.patient_id, ctdna_results
        )
        
        return MRDAnalysisResult(
            sample_id=biopsy_data.sample_id,
            collection_date=biopsy_data.collection_date,
            ctdna_detected=ctdna_results['detected'],
            ctdna_concentration=ctdna_results['concentration'],
            variant_allele_frequency=ctdna_results['vaf'],
            tumor_fraction=ctdna_results['tumor_fraction'],
            mrd_status=mrd_status,
            detection_sensitivity=ctdna_results['sensitivity'],
            tracking_mutations=mutation_analysis['tracking_mutations'],
            clonal_evolution=clonal_evolution,
            treatment_response=self._assess_treatment_response(biopsy_data.patient_id),
            relapse_risk=relapse_risk,
            monitoring_recommendations=recommendations,
            longitudinal_trend=longitudinal_trend,
            confidence_score=ctdna_results['confidence'],
            analysis_timestamp=datetime.now(),
            technical_quality=qc_results['metrics']
        )
    
    def _perform_quality_control(self, biopsy_data: LiquidBiopsyData) -> Dict:
        """Perform quality control on liquid biopsy sample"""
        
        qc_metrics = {}
        passed = True
        
        # Check cfDNA concentration
        qc_metrics['cfDNA_concentration'] = biopsy_data.cfDNA_concentration
        if biopsy_data.cfDNA_concentration < 0.1:  # ng/mL
            passed = False
        
        # Check cfDNA integrity
        qc_metrics['cfDNA_integrity'] = biopsy_data.cfDNA_integrity
        if biopsy_data.cfDNA_integrity < 0.7:
            passed = False
        
        # Check white cell contamination
        qc_metrics['white_cell_contamination'] = biopsy_data.white_cell_contamination
        if biopsy_data.white_cell_contamination > 0.1:
            passed = False
        
        # Check processing delay
        qc_metrics['processing_delay'] = biopsy_data.processing_delay_hours
        if biopsy_data.processing_delay_hours > 24:
            passed = False
        
        # Check sample volume
        qc_metrics['sample_volume'] = biopsy_data.volume_ml
        if biopsy_data.volume_ml < 1.0:
            passed = False
        
        return {
            'passed': passed,
            'metrics': qc_metrics
        }
    
    def _detect_ctdna(self, biopsy_data: LiquidBiopsyData) -> Dict:
        """Detect and quantify circulating tumor DNA"""
        
        # Simulate ctDNA detection using mutations
        mutations = biopsy_data.mutations_detected
        
        if not mutations:
            return {
                'detected': False,
                'concentration': 0.0,
                'vaf': 0.0,
                'tumor_fraction': 0.0,
                'sensitivity': self.detection_thresholds['standard'],
                'confidence': 0.95
            }
        
        # Calculate aggregate VAF
        total_vaf = sum(mut.get('vaf', 0) for mut in mutations)
        avg_vaf = total_vaf / len(mutations) if mutations else 0
        
        # Estimate tumor fraction
        tumor_fraction = min(avg_vaf * 2, 1.0)  # Diploid assumption
        
        # Calculate ctDNA concentration
        ctdna_concentration = tumor_fraction * biopsy_data.cfDNA_concentration
        
        # Determine detection status
        detected = avg_vaf > self.detection_thresholds['standard']
        
        # Calculate confidence based on coverage and VAF
        confidence = min(0.95, 0.5 + (avg_vaf * 10))
        
        return {
            'detected': detected,
            'concentration': ctdna_concentration,
            'vaf': avg_vaf,
            'tumor_fraction': tumor_fraction,
            'sensitivity': self.detection_thresholds['standard'],
            'confidence': confidence
        }
    
    def _analyze_mutations(self, mutations: List[Dict]) -> Dict:
        """Analyze mutation patterns for tracking"""
        
        tracking_mutations = []
        actionable_mutations = []
        resistance_mutations = []
        
        for mutation in mutations:
            mut_info = {
                'gene': mutation.get('gene', 'Unknown'),
                'variant': mutation.get('variant', 'Unknown'),
                'vaf': mutation.get('vaf', 0),
                'coverage': mutation.get('coverage', 0),
                'quality_score': mutation.get('quality', 0),
                'tracking_priority': 'high' if mutation.get('vaf', 0) > 0.01 else 'low'
            }
            
            tracking_mutations.append(mut_info)
            
            # Categorize mutations
            if mutation.get('actionable', False):
                actionable_mutations.append(mut_info)
            
            if mutation.get('resistance', False):
                resistance_mutations.append(mut_info)
        
        return {
            'tracking_mutations': tracking_mutations,
            'actionable_mutations': actionable_mutations,
            'resistance_mutations': resistance_mutations,
            'mutation_count': len(mutations),
            'high_confidence_count': len([m for m in mutations if m.get('vaf', 0) > 0.01])
        }
    
    def _assess_clonal_evolution(self, patient_id: str, current_mutations: List[Dict]) -> Dict[str, float]:
        """Assess clonal evolution patterns"""
        
        # Store current mutations
        self.mutation_tracker[patient_id].append({
            'timestamp': datetime.now(),
            'mutations': current_mutations
        })
        
        evolution_metrics = {}
        
        # Get historical data
        historical_data = self.mutation_tracker[patient_id]
        
        if len(historical_data) < 2:
            evolution_metrics['stability'] = 1.0
            evolution_metrics['diversity'] = 0.0
            evolution_metrics['emergence_rate'] = 0.0
            return evolution_metrics
        
        # Calculate clonal stability
        current_genes = set(mut.get('gene', '') for mut in current_mutations)
        prev_genes = set(mut.get('gene', '') for mut in historical_data[-2]['mutations'])
        
        overlap = len(current_genes.intersection(prev_genes))
        union = len(current_genes.union(prev_genes))
        stability = overlap / union if union > 0 else 1.0
        
        # Calculate clonal diversity (Shannon entropy)
        vafs = [mut.get('vaf', 0) for mut in current_mutations if mut.get('vaf', 0) > 0]
        if vafs:
            norm_vafs = np.array(vafs) / sum(vafs)
            diversity = -sum(p * np.log2(p) for p in norm_vafs if p > 0)
        else:
            diversity = 0.0
        
        # Calculate emergence rate
        new_mutations = current_genes - prev_genes
        emergence_rate = len(new_mutations) / len(current_genes) if current_genes else 0.0
        
        evolution_metrics.update({
            'stability': stability,
            'diversity': diversity,
            'emergence_rate': emergence_rate,
            'new_mutations': len(new_mutations),
            'lost_mutations': len(prev_genes - current_genes)
        })
        
        return evolution_metrics
    
    def _determine_mrd_status(self, ctdna_results: Dict, mutation_analysis: Dict) -> str:
        """Determine overall MRD status"""
        
        if not ctdna_results['detected']:
            return "Negative"
        
        # Check for high-confidence mutations
        high_conf_count = mutation_analysis['high_confidence_count']
        avg_vaf = ctdna_results['vaf']
        
        if high_conf_count >= 2 and avg_vaf > 0.01:
            return "Positive"
        elif high_conf_count >= 1 and avg_vaf > 0.005:
            return "Positive"
        elif ctdna_results['confidence'] < 0.8:
            return "Indeterminate"
        else:
            return "Negative"
    
    def _calculate_relapse_risk(self, ctdna_results: Dict, 
                               mutation_analysis: Dict, 
                               clonal_evolution: Dict) -> float:
        """Calculate risk of disease relapse"""
        
        base_risk = 0.1  # 10% baseline
        
        # ctDNA detection increases risk
        if ctdna_results['detected']:
            base_risk += ctdna_results['vaf'] * 20  # Scale VAF to risk
        
        # High mutation burden increases risk
        mutation_count = mutation_analysis['mutation_count']
        base_risk += min(mutation_count * 0.05, 0.3)
        
        # Clonal instability increases risk
        if clonal_evolution.get('stability', 1.0) < 0.7:
            base_risk += 0.2
        
        # High emergence rate increases risk
        emergence_rate = clonal_evolution.get('emergence_rate', 0.0)
        base_risk += emergence_rate * 0.3
        
        return min(base_risk, 0.95)  # Cap at 95%
    
    def _generate_monitoring_recommendations(self, mrd_status: str, 
                                           relapse_risk: float, 
                                           biopsy_data: LiquidBiopsyData) -> List[str]:
        """Generate personalized monitoring recommendations"""
        
        recommendations = []
        
        if mrd_status == "Positive":
            recommendations.extend([
                "Increase monitoring frequency to every 4-6 weeks",
                "Consider imaging studies to assess disease burden",
                "Evaluate for treatment modification or intensification",
                "Monitor for emergence of resistance mutations"
            ])
        elif mrd_status == "Negative":
            if relapse_risk > 0.3:
                recommendations.extend([
                    "Continue monitoring every 8-12 weeks",
                    "Maintain current treatment regimen",
                    "Consider extended monitoring period"
                ])
            else:
                recommendations.extend([
                    "Reduce monitoring frequency to every 12-16 weeks",
                    "Consider treatment de-escalation if appropriate"
                ])
        else:  # Indeterminate
            recommendations.extend([
                "Repeat testing in 2-4 weeks",
                "Improve sample quality for next collection",
                "Consider alternative monitoring methods",
                "Maintain current treatment until clarification"
            ])
        
        # Add technical recommendations
        if biopsy_data.cfDNA_concentration < 1.0:
            recommendations.append("Increase sample volume for future collections")
        
        if biopsy_data.processing_delay_hours > 6:
            recommendations.append("Reduce sample processing delay time")
        
        return recommendations
    
    def _assess_longitudinal_trends(self, patient_id: str, ctdna_results: Dict) -> str:
        """Assess longitudinal trends in ctDNA levels"""
        
        # Store current results
        self.longitudinal_data[patient_id].append({
            'timestamp': datetime.now(),
            'vaf': ctdna_results['vaf'],
            'concentration': ctdna_results['concentration'],
            'detected': ctdna_results['detected']
        })
        
        data = self.longitudinal_data[patient_id]
        
        if len(data) < 2:
            return "Insufficient data"
        
        # Compare recent trends
        recent_vafs = [d['vaf'] for d in data[-3:]]  # Last 3 measurements
        
        if len(recent_vafs) >= 2:
            if all(recent_vafs[i] < recent_vafs[i-1] for i in range(1, len(recent_vafs))):
                return "Decreasing"
            elif all(recent_vafs[i] > recent_vafs[i-1] for i in range(1, len(recent_vafs))):
                return "Increasing"
            else:
                return "Stable"
        
        return "Stable"
    
    def _assess_treatment_response(self, patient_id: str) -> str:
        """Assess treatment response based on longitudinal data"""
        
        data = self.longitudinal_data[patient_id]
        
        if len(data) < 2:
            return "Cannot assess"
        
        # Compare first and last measurements
        initial_vaf = data[0]['vaf']
        current_vaf = data[-1]['vaf']
        
        reduction = (initial_vaf - current_vaf) / initial_vaf if initial_vaf > 0 else 0
        
        if reduction > 0.5:
            return "Good response"
        elif reduction > 0.2:
            return "Partial response"
        elif reduction > -0.2:
            return "Stable disease"
        else:
            return "Progressive disease"
    
    def _create_failed_result(self, biopsy_data: LiquidBiopsyData, qc_results: Dict) -> MRDAnalysisResult:
        """Create result for failed quality control"""
        
        return MRDAnalysisResult(
            sample_id=biopsy_data.sample_id,
            collection_date=biopsy_data.collection_date,
            ctdna_detected=False,
            ctdna_concentration=0.0,
            variant_allele_frequency=0.0,
            tumor_fraction=0.0,
            mrd_status="Quality Control Failed",
            detection_sensitivity=0.0,
            tracking_mutations=[],
            clonal_evolution={},
            treatment_response="Cannot assess",
            relapse_risk=0.0,
            monitoring_recommendations=[
                "Improve sample collection procedures",
                "Repeat testing with better quality sample"
            ],
            longitudinal_trend="Cannot assess",
            confidence_score=0.0,
            analysis_timestamp=datetime.now(),
            technical_quality=qc_results['metrics']
        )

def create_sample_liquid_biopsy() -> LiquidBiopsyData:
    """Create sample liquid biopsy data for testing"""
    
    sample_mutations = [
        {
            'gene': 'EGFR',
            'variant': 'L858R',
            'vaf': 0.025,
            'coverage': 1500,
            'quality': 40,
            'actionable': True,
            'resistance': False
        },
        {
            'gene': 'TP53',
            'variant': 'R273H',
            'vaf': 0.018,
            'coverage': 1200,
            'quality': 38,
            'actionable': False,
            'resistance': False
        },
        {
            'gene': 'KRAS',
            'variant': 'G12C',
            'vaf': 0.012,
            'coverage': 1800,
            'quality': 42,
            'actionable': True,
            'resistance': False
        }
    ]
    
    return LiquidBiopsyData(
        sample_id="LB_001_20250920",
        patient_id="PT_001",
        collection_date=datetime.now(),
        sample_type="plasma",
        volume_ml=10.0,
        processing_delay_hours=2.5,
        storage_conditions="-80C",
        cfDNA_concentration=2.5,  # ng/mL
        cfDNA_integrity=0.85,
        white_cell_contamination=0.02,
        mutations_detected=sample_mutations,
        copy_number_variants=[],
        methylation_markers=[],
        protein_biomarkers={
            'CEA': 5.2,
            'CA19-9': 15.8,
            'PSA': 2.1
        }
    )

def run_mrd_analysis_demo():
    """Run a demonstration of MRD analysis"""
    
    print("=" * 60)
    print("MRD DETECTION ANALYSIS DEMONSTRATION")
    print("=" * 60)
    
    # Create pipeline
    mrd_pipeline = MRDDetectionPipeline()
    
    # Create sample data
    biopsy_data = create_sample_liquid_biopsy()
    
    print(f"Analyzing sample: {biopsy_data.sample_id}")
    print(f"Patient ID: {biopsy_data.patient_id}")
    print(f"Collection date: {biopsy_data.collection_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Sample type: {biopsy_data.sample_type}")
    print(f"cfDNA concentration: {biopsy_data.cfDNA_concentration:.2f} ng/mL")
    print(f"Mutations detected: {len(biopsy_data.mutations_detected)}")
    print()
    
    # Run analysis
    result = mrd_pipeline.analyze_liquid_biopsy(biopsy_data)
    
    # Display results
    print("MRD ANALYSIS RESULTS:")
    print(f"ctDNA Detected: {'Yes' if result.ctdna_detected else 'No'}")
    print(f"MRD Status: {result.mrd_status}")
    print(f"ctDNA Concentration: {result.ctdna_concentration:.4f} copies/mL")
    print(f"Variant Allele Frequency: {result.variant_allele_frequency:.4f}")
    print(f"Tumor Fraction: {result.tumor_fraction:.4f}")
    print(f"Detection Sensitivity: {result.detection_sensitivity:.0e}")
    print(f"Relapse Risk: {result.relapse_risk:.1%}")
    print(f"Treatment Response: {result.treatment_response}")
    print(f"Longitudinal Trend: {result.longitudinal_trend}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    print()
    
    print("TRACKING MUTATIONS:")
    for i, mutation in enumerate(result.tracking_mutations[:5], 1):
        print(f"{i}. {mutation['gene']} {mutation['variant']} (VAF: {mutation['vaf']:.4f})")
    print()
    
    print("MONITORING RECOMMENDATIONS:")
    for i, rec in enumerate(result.monitoring_recommendations, 1):
        print(f"{i}. {rec}")
    print()
    
    print("CLONAL EVOLUTION METRICS:")
    for metric, value in result.clonal_evolution.items():
        print(f"{metric.title()}: {value:.3f}")
    print()
    
    return result

if __name__ == "__main__":
    # Run demonstration
    result = run_mrd_analysis_demo()
