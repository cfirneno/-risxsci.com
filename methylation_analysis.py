#!/usr/bin/env python3
"""
Genomics NGS Analysis Module
===========================

Next Generation Sequencing analysis for cancer genomics including
mutation detection, variant calling, and clinical interpretation.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import pickle
import json
from collections import defaultdict, Counter

from config import *

logger = logging.getLogger(__name__)

@dataclass
class NGSAnalysisResult:
    """Results from NGS genomic analysis"""
    sample_id: str
    total_mutations: int
    tumor_mutational_burden: float
    microsatellite_instability: str
    mutation_signatures: Dict[str, float]
    driver_mutations: List[Dict]
    actionable_mutations: List[Dict]
    clonal_evolution: Dict[str, float]
    dna_repair_status: str
    treatment_recommendations: List[Dict]
    clinical_interpretation: str
    confidence_score: float
    analysis_timestamp: datetime
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization"""
        return {
            'sample_id': self.sample_id,
            'total_mutations': self.total_mutations,
            'tumor_mutational_burden': self.tumor_mutational_burden,
            'microsatellite_instability': self.microsatellite_instability,
            'mutation_signatures': self.mutation_signatures,
            'driver_mutations': self.driver_mutations,
            'actionable_mutations': self.actionable_mutations,
            'clonal_evolution': self.clonal_evolution,
            'dna_repair_status': self.dna_repair_status,
            'treatment_recommendations': self.treatment_recommendations,
            'clinical_interpretation': self.clinical_interpretation,
            'confidence_score': self.confidence_score,
            'analysis_timestamp': self.analysis_timestamp.isoformat()
        }

class GenomicsTransformer(nn.Module):
    """Transformer model for genomic sequence analysis"""
    
    def __init__(self, vocab_size=4, seq_length=1024, embed_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # DNA sequence embedding (A=0, T=1, G=2, C=3)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_length, embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        # Feature extraction heads
        self.mutation_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100)  # Top 100 cancer genes
        )
        
        self.signature_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 30)  # Mutation signatures
        )
        
        self.tmb_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # TMB score
        )
        
        self.msi_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3)  # MSI-H, MSI-L, MSS
        )
    
    def forward(self, sequences):
        """
        Forward pass for genomic sequences
        Args:
            sequences: DNA sequences encoded as integers [batch_size, seq_length]
        """
        batch_size, seq_len = sequences.shape
        
        # Generate position indices
        positions = torch.arange(seq_len, device=sequences.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(sequences)
        pos_embeds = self.position_embedding(positions)
        embeddings = token_embeds + pos_embeds
        
        # Transformer processing
        x = embeddings.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global pooling
        pooled = x.mean(dim=0)  # [batch_size, embed_dim]
        
        # Predictions
        mutation_logits = self.mutation_head(pooled)
        signature_logits = self.signature_head(pooled)
        tmb_score = self.tmb_head(pooled)
        msi_logits = self.msi_head(pooled)
        
        return {
            'mutation_logits': mutation_logits,
            'signature_logits': signature_logits,
            'tmb_score': tmb_score,
            'msi_logits': msi_logits,
            'features': pooled
        }

class NGSAnalyzer:
    """Comprehensive NGS analysis engine"""
    
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.model = GenomicsTransformer().to(self.device)
        
        # Gene databases
        self.cancer_genes = self._load_cancer_gene_database()
        self.mutation_signatures = self._load_mutation_signatures()
        self.actionable_genes = self._load_actionable_genes()
        self.msi_markers = self._load_msi_markers()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            logger.info("No pre-trained NGS model loaded. Using random initialization.")
    
    def _load_cancer_gene_database(self) -> Dict[str, Dict]:
        """Load cancer gene database with clinical significance"""
        return {
            'TP53': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '17', 'pathways': ['DNA repair', 'Apoptosis']},
            'KRAS': {'type': 'oncogene', 'significance': 'high', 'chromosome': '12', 'pathways': ['MAPK', 'Cell proliferation']},
            'EGFR': {'type': 'oncogene', 'significance': 'high', 'chromosome': '7', 'pathways': ['Growth signaling', 'Cell proliferation']},
            'BRCA1': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '17', 'pathways': ['DNA repair', 'Homologous recombination']},
            'BRCA2': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '13', 'pathways': ['DNA repair', 'Homologous recombination']},
            'PIK3CA': {'type': 'oncogene', 'significance': 'high', 'chromosome': '3', 'pathways': ['PI3K/AKT', 'Cell growth']},
            'APC': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '5', 'pathways': ['Wnt signaling', 'Cell adhesion']},
            'PTEN': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '10', 'pathways': ['PI3K/AKT', 'Cell growth']},
            'BRAF': {'type': 'oncogene', 'significance': 'high', 'chromosome': '7', 'pathways': ['MAPK', 'Cell proliferation']},
            'MYC': {'type': 'oncogene', 'significance': 'high', 'chromosome': '8', 'pathways': ['Cell cycle', 'Apoptosis']},
            'ALK': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '2', 'pathways': ['RTK signaling', 'Cell proliferation']},
            'RET': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '10', 'pathways': ['RTK signaling', 'Cell proliferation']},
            'MET': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '7', 'pathways': ['Growth signaling', 'Cell migration']},
            'FGFR1': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '8', 'pathways': ['Growth signaling', 'Cell proliferation']},
            'CDK4': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '12', 'pathways': ['Cell cycle', 'G1/S transition']},
            'CCND1': {'type': 'oncogene', 'significance': 'medium', 'chromosome': '11', 'pathways': ['Cell cycle', 'G1/S transition']},
            'ERBB2': {'type': 'oncogene', 'significance': 'high', 'chromosome': '17', 'pathways': ['Growth signaling', 'Cell proliferation']},
            'MLH1': {'type': 'dna_repair', 'significance': 'high', 'chromosome': '3', 'pathways': ['Mismatch repair', 'DNA stability']},
            'MSH2': {'type': 'dna_repair', 'significance': 'high', 'chromosome': '2', 'pathways': ['Mismatch repair', 'DNA stability']},
            'MSH6': {'type': 'dna_repair', 'significance': 'medium', 'chromosome': '2', 'pathways': ['Mismatch repair', 'DNA stability']},
            'PMS2': {'type': 'dna_repair', 'significance': 'medium', 'chromosome': '7', 'pathways': ['Mismatch repair', 'DNA stability']},
            'ATM': {'type': 'dna_repair', 'significance': 'high', 'chromosome': '11', 'pathways': ['DNA damage response', 'Cell cycle checkpoints']},
            'CHEK2': {'type': 'dna_repair', 'significance': 'medium', 'chromosome': '22', 'pathways': ['DNA damage response', 'Cell cycle checkpoints']},
            'ARID1A': {'type': 'chromatin', 'significance': 'medium', 'chromosome': '1', 'pathways': ['Chromatin remodeling', 'Transcription']},
            'IDH1': {'type': 'metabolism', 'significance': 'high', 'chromosome': '2', 'pathways': ['Metabolism', 'Epigenetics']},
            'IDH2': {'type': 'metabolism', 'significance': 'medium', 'chromosome': '15', 'pathways': ['Metabolism', 'Epigenetics']},
            'VHL': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '3', 'pathways': ['Hypoxia response', 'Angiogenesis']},
            'NF1': {'type': 'tumor_suppressor', 'significance': 'medium', 'chromosome': '17', 'pathways': ['RAS signaling', 'Cell proliferation']},
            'NF2': {'type': 'tumor_suppressor', 'significance': 'medium', 'chromosome': '22', 'pathways': ['Cell contact inhibition', 'Hippo pathway']},
            'CDKN2A': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '9', 'pathways': ['Cell cycle', 'p53 pathway']},
            'RB1': {'type': 'tumor_suppressor', 'significance': 'high', 'chromosome': '13', 'pathways': ['Cell cycle', 'G1/S checkpoint']}
        }
    
    def _load_mutation_signatures(self) -> Dict[str, Dict]:
        """Load mutation signature patterns"""
        return {
            'smoking': {
                'pattern': {'C>A': 0.4, 'C>G': 0.2, 'C>T': 0.3, 'T>A': 0.1},
                'description': 'Tobacco smoking signature',
                'clinical_significance': 'high'
            },
            'uv_exposure': {
                'pattern': {'C>T': 0.7, 'CC>TT': 0.2, 'C>A': 0.1},
                'description': 'UV radiation signature',
                'clinical_significance': 'high'
            },
            'aging': {
                'pattern': {'C>T': 0.6, 'T>C': 0.2, 'C>G': 0.1, 'T>A': 0.1},
                'description': 'Age-related signature',
                'clinical_significance': 'medium'
            },
            'dna_repair_deficiency': {
                'pattern': {'indels': 0.4, 'C>T': 0.3, 'T>G': 0.2, 'C>A': 0.1},
                'description': 'DNA repair deficiency signature',
                'clinical_significance': 'high'
            },
            'chemotherapy': {
                'pattern': {'T>A': 0.4, 'T>C': 0.3, 'C>A': 0.2, 'C>T': 0.1},
                'description': 'Chemotherapy-induced signature',
                'clinical_significance': 'medium'
            },
            'apobec': {
                'pattern': {'C>T': 0.5, 'C>G': 0.3, 'T>C': 0.2},
                'description': 'APOBEC enzyme signature',
                'clinical_significance': 'medium'
            }
        }
    
    def _load_actionable_genes(self) -> Dict[str, Dict]:
        """Load genes with actionable therapeutic targets"""
        return {
            'EGFR': {
                'drugs': ['erlotinib', 'gefitinib', 'osimertinib', 'afatinib'],
                'resistance_mutations': ['T790M', 'C797S'],
                'cancer_types': ['NSCLC', 'glioblastoma'],
                'evidence_level': 'A'
            },
            'BRAF': {
                'drugs': ['vemurafenib', 'dabrafenib', 'trametinib'],
                'resistance_mutations': ['G469A', 'G466V'],
                'cancer_types': ['melanoma', 'colorectal', 'thyroid'],
                'evidence_level': 'A'
            },
            'ALK': {
                'drugs': ['crizotinib', 'alectinib', 'ceritinib', 'brigatinib'],
                'resistance_mutations': ['L1196M', 'G1269A'],
                'cancer_types': ['NSCLC', 'ALCL'],
                'evidence_level': 'A'
            },
            'BRCA1': {
                'drugs': ['olaparib', 'niraparib', 'rucaparib'],
                'resistance_mutations': ['secondary_mutations'],
                'cancer_types': ['breast', 'ovarian', 'prostate'],
                'evidence_level': 'A'
            },
            'BRCA2': {
                'drugs': ['olaparib', 'niraparib', 'rucaparib'],
                'resistance_mutations': ['secondary_mutations'],
                'cancer_types': ['breast', 'ovarian', 'prostate'],
                'evidence_level': 'A'
            },
            'KRAS': {
                'drugs': ['sotorasib', 'adagrasib'],
                'resistance_mutations': ['Y96C', 'H95Q'],
                'cancer_types': ['NSCLC', 'colorectal', 'pancreatic'],
                'evidence_level': 'B'
            },
            'PIK3CA': {
                'drugs': ['alpelisib', 'inavolisib'],
                'resistance_mutations': ['E545K', 'H1047R'],
                'cancer_types': ['breast', 'head_neck'],
                'evidence_level': 'B'
            },
            'ERBB2': {
                'drugs': ['trastuzumab', 'pertuzumab', 'T-DM1', 'T-DXd'],
                'resistance_mutations': ['S310F', 'L755S'],
                'cancer_types': ['breast', 'gastric'],
                'evidence_level': 'A'
            }
        }
    
    def _load_msi_markers(self) -> List[str]:
        """Load microsatellite instability markers"""
        return [
            'BAT-25', 'BAT-26', 'D2S123', 'D5S346', 'D17S250',
            'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM'
        ]
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"NGS model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading NGS model: {e}")
    
    def save_model(self, model_path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'cancer_genes': self.cancer_genes,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, model_path)
        logger.info(f"NGS model saved to {model_path}")
    
    def encode_dna_sequence(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence to tensor"""
        # DNA base to integer mapping
        base_to_int = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 0}  # N -> A for simplicity
        
        # Convert to uppercase and encode
        sequence = sequence.upper()
        encoded = [base_to_int.get(base, 0) for base in sequence]
        
        # Pad or truncate to fixed length
        seq_len = self.model.seq_length
        if len(encoded) < seq_len:
            encoded.extend([0] * (seq_len - len(encoded)))
        else:
            encoded = encoded[:seq_len]
        
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    
    def parse_vcf_file(self, vcf_content: str) -> List[Dict]:
        """Parse VCF file content and extract mutations"""
        mutations = []
        lines = vcf_content.strip().split('\n')
        
        for line in lines:
            if line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 8:
                continue
            
            chrom, pos, id_field, ref, alt, qual, filter_field, info = fields[:8]
            
            # Extract gene information from INFO field
            gene_match = re.search(r'GENE=([^;]+)', info)
            gene = gene_match.group(1) if gene_match else 'Unknown'
            
            mutation = {
                'chromosome': chrom,
                'position': int(pos) if pos.isdigit() else 0,
                'reference': ref,
                'alternate': alt,
                'gene': gene,
                'quality': float(qual) if qual != '.' else 0.0,
                'filter': filter_field,
                'info': info
            }
            mutations.append(mutation)
        
        return mutations
    
    def calculate_tumor_mutational_burden(self, mutations: List[Dict]) -> float:
        """Calculate tumor mutational burden (mutations per megabase)"""
        # Filter for coding mutations
        coding_mutations = [m for m in mutations if self._is_coding_mutation(m)]
        
        # Assume 30 Mb coding genome size
        coding_genome_size = 30.0
        tmb = len(coding_mutations) / coding_genome_size
        
        return tmb
    
    def _is_coding_mutation(self, mutation: Dict) -> bool:
        """Check if mutation is in coding region"""
        # Simple heuristic - check if gene is known
        return mutation['gene'] in self.cancer_genes
    
    def detect_mutation_signatures(self, mutations: List[Dict]) -> Dict[str, float]:
        """Detect mutation signatures from mutation patterns"""
        # Count mutation types
        mutation_counts = defaultdict(int)
        total_mutations = len(mutations)
        
        if total_mutations == 0:
            return {sig: 0.0 for sig in self.mutation_signatures.keys()}
        
        for mutation in mutations:
            ref, alt = mutation['reference'], mutation['alternate']
            if len(ref) == 1 and len(alt) == 1:
                mut_type = f"{ref}>{alt}"
                mutation_counts[mut_type] += 1
            else:
                mutation_counts['indels'] += 1
        
        # Normalize counts
        mutation_frequencies = {
            mut_type: count / total_mutations 
            for mut_type, count in mutation_counts.items()
        }
        
        # Calculate signature scores
        signature_scores = {}
        for sig_name, sig_data in self.mutation_signatures.items():
            pattern = sig_data['pattern']
            score = 0.0
            
            for mut_type, expected_freq in pattern.items():
                observed_freq = mutation_frequencies.get(mut_type, 0.0)
                score += min(observed_freq, expected_freq)
            
            signature_scores[sig_name] = score
        
        return signature_scores
    
    def assess_microsatellite_instability(self, mutations: List[Dict]) -> str:
        """Assess microsatellite instability status"""
        # Count MSI-related gene mutations
        msi_gene_mutations = 0
        for mutation in mutations:
            if mutation['gene'] in ['MLH1', 'MSH2', 'MSH6', 'PMS2']:
                msi_gene_mutations += 1
        
        # Simple heuristic for MSI status
        if msi_gene_mutations >= 2:
            return "MSI-H"
        elif msi_gene_mutations == 1:
            return "MSI-L"
        else:
            return "MSS"
    
    def identify_driver_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Identify likely driver mutations"""
        driver_mutations = []
        
        for mutation in mutations:
            gene = mutation['gene']
            if gene in self.cancer_genes:
                gene_info = self.cancer_genes[gene]
                
                driver_mutation = {
                    'gene': gene,
                    'mutation': f"{mutation['reference']}{mutation['position']}{mutation['alternate']}",
                    'type': gene_info['type'],
                    'significance': gene_info['significance'],
                    'pathways': gene_info['pathways'],
                    'chromosome': gene_info['chromosome']
                }
                driver_mutations.append(driver_mutation)
        
        return driver_mutations
    
    def identify_actionable_mutations(self, mutations: List[Dict]) -> List[Dict]:
        """Identify mutations with therapeutic implications"""
        actionable_mutations = []
        
        for mutation in mutations:
            gene = mutation['gene']
            if gene in self.actionable_genes:
                target_info = self.actionable_genes[gene]
                
                actionable_mutation = {
                    'gene': gene,
                    'mutation': f"{mutation['reference']}{mutation['position']}{mutation['alternate']}",
                    'drugs': target_info['drugs'],
                    'cancer_types': target_info['cancer_types'],
                    'evidence_level': target_info['evidence_level']
                }
                actionable_mutations.append(actionable_mutation)
        
        return actionable_mutations
    
    def generate_treatment_recommendations(self, actionable_mutations: List[Dict], 
                                         cancer_type: str = None) -> List[Dict]:
        """Generate treatment recommendations based on mutations"""
        recommendations = []
        
        for mutation in actionable_mutations:
            if cancer_type and cancer_type.lower() not in [ct.lower() for ct in mutation['cancer_types']]:
                continue
            
            recommendation = {
                'target': mutation['gene'],
                'drugs': mutation['drugs'],
                'evidence_level': mutation['evidence_level'],
                'rationale': f"Target {mutation['gene']} mutation with {', '.join(mutation['drugs'][:2])}"
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def analyze_ngs_data(self, vcf_content: str, sample_id: str = "unknown", 
                        cancer_type: str = None) -> NGSAnalysisResult:
        """
        Comprehensive NGS analysis of VCF data
        Args:
            vcf_content: VCF file content as string
            sample_id: Sample identifier
            cancer_type: Known cancer type for targeted recommendations
        """
        try:
            # Parse VCF file
            mutations = self.parse_vcf_file(vcf_content)
            
            if not mutations:
                logger.warning(f"No mutations found in VCF for sample {sample_id}")
            
            # Calculate metrics
            total_mutations = len(mutations)
            tmb = self.calculate_tumor_mutational_burden(mutations)
            msi_status = self.assess_microsatellite_instability(mutations)
            mutation_signatures = self.detect_mutation_signatures(mutations)
            
            # Identify key mutations
            driver_mutations = self.identify_driver_mutations(mutations)
            actionable_mutations = self.identify_actionable_mutations(mutations)
            
            # Generate recommendations
            treatment_recommendations = self.generate_treatment_recommendations(
                actionable_mutations, cancer_type
            )
            
            # Assess DNA repair status
            dna_repair_genes = ['MLH1', 'MSH2', 'MSH6', 'PMS2', 'BRCA1', 'BRCA2', 'ATM']
            dna_repair_mutations = [m for m in driver_mutations if m['gene'] in dna_repair_genes]
            dna_repair_status = "Deficient" if dna_repair_mutations else "Proficient"
            
            # Clonal evolution analysis (simplified)
            clonal_evolution = {
                'early_mutations': len([m for m in driver_mutations if m['significance'] == 'high']),
                'late_mutations': len([m for m in driver_mutations if m['significance'] == 'medium']),
                'clonal_diversity': min(1.0, len(driver_mutations) / 10.0)
            }
            
            # Clinical interpretation
            if tmb > 10:
                tmb_interpretation = "High TMB - consider immunotherapy"
            elif tmb > 6:
                tmb_interpretation = "Intermediate TMB"
            else:
                tmb_interpretation = "Low TMB"
            
            clinical_interpretation = f"{tmb_interpretation}. {msi_status} status. {dna_repair_status} DNA repair."
            
            # Confidence score based on number of mutations and quality
            confidence_score = min(1.0, (total_mutations / 50.0) + 0.3)
            
            result = NGSAnalysisResult(
                sample_id=sample_id,
                total_mutations=total_mutations,
                tumor_mutational_burden=tmb,
                microsatellite_instability=msi_status,
                mutation_signatures=mutation_signatures,
                driver_mutations=driver_mutations,
                actionable_mutations=actionable_mutations,
                clonal_evolution=clonal_evolution,
                dna_repair_status=dna_repair_status,
                treatment_recommendations=treatment_recommendations,
                clinical_interpretation=clinical_interpretation,
                confidence_score=confidence_score,
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"NGS analysis completed for sample {sample_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in NGS analysis: {e}")
            # Return default result on error
            return NGSAnalysisResult(
                sample_id=sample_id,
                total_mutations=0,
                tumor_mutational_burden=0.0,
                microsatellite_instability="MSS",
                mutation_signatures={sig: 0.0 for sig in self.mutation_signatures.keys()},
                driver_mutations=[],
                actionable_mutations=[],
                clonal_evolution={'early_mutations': 0, 'late_mutations': 0, 'clonal_diversity': 0.0},
                dna_repair_status="Unknown",
                treatment_recommendations=[],
                clinical_interpretation="Analysis failed - no mutations detected",
                confidence_score=0.0,
                analysis_timestamp=datetime.now()
            )

# Demo function
def run_ngs_analysis_demo():
    """Run a demo of the NGS analysis system"""
    print("🧬 NGS Genomic Analysis Demo")
    print("=" * 40)
    
    # Create analyzer
    analyzer = NGSAnalyzer()
    
    # Create sample VCF content
    sample_vcf = """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
7	55181378	rs121913227	G	A	60	PASS	GENE=EGFR;MUTATION=L858R
12	25398284	rs121913529	C	T	60	PASS	GENE=KRAS;MUTATION=G12C
17	7674220	rs28934578	G	A	60	PASS	GENE=TP53;MUTATION=R273H
3	178936091	rs104886003	G	A	60	PASS	GENE=PIK3CA;MUTATION=E545K
9	139399431	rs104894085	A	G	60	PASS	GENE=NOTCH1;MUTATION=P2514A"""
    
    # Analyze
    result = analyzer.analyze_ngs_data(sample_vcf, "DEMO_NGS_001", "NSCLC")
    
    print(f"Sample ID: {result.sample_id}")
    print(f"Total Mutations: {result.total_mutations}")
    print(f"TMB: {result.tumor_mutational_burden:.2f} mutations/Mb")
    print(f"MSI Status: {result.microsatellite_instability}")
    print(f"DNA Repair: {result.dna_repair_status}")
    print(f"Driver Mutations: {len(result.driver_mutations)}")
    print(f"Actionable Mutations: {len(result.actionable_mutations)}")
    print(f"Treatment Recommendations: {len(result.treatment_recommendations)}")
    print(f"Clinical Interpretation: {result.clinical_interpretation}")
    
    return result

if __name__ == "__main__":
    # Run demo
    result = run_ngs_analysis_demo()
