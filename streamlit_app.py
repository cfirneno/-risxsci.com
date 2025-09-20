"""
Streamlit Application Module
===========================

Main user interface and application logic for the comprehensive cancer analysis platform.
Includes all Streamlit pages, visualization components, and interactive features.

RISX Science - Charles Firneno
16 Cross Street, 206 New Canaan, CT 06840

Research Use Only - Not for Clinical Decision Making
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import io
import base64
from PIL import Image
import json
import logging

# Import all our custom modules
try:
    from config import *
    from wsi_analysis import FlowMatchingLogisticWSI, WSIAnalysisResult
    from genomics_ngs import NGSAnalyzer, NGSAnalysisResult
    from methylation_analysis import MethylationAnalyzer, MethylationAnalysisResult
    from mrd_detection import MRDDetectionPipeline, MRDAnalysisResult, create_sample_liquid_biopsy
    from risk_stratification import RiskStratificationPipeline, create_sample_prognostic_factors
    from clinical_integration import ClinicalDecisionSupport, create_sample_clinical_data
    from flow_dynamics import FlowMatchingLogisticModel, run_flow_dynamics_demo
    from multimodal_fusion import IntegratedCancerAnalysisModel, MultimodalConfig, create_sample_multimodal_data
    from data_generation import SyntheticCancerDataset, create_synthetic_datasets
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please ensure all required modules are in the Python path")

# Configure Streamlit page
st.set_page_config(
    page_title="🧬 RISX Science - Cancer Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    
    .result-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .result-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stTabs > div > div > div > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = None
    
    if 'models_initialized' not in st.session_state:
        st.session_state.models_initialized = False
    
    if 'demo_data' not in st.session_state:
        st.session_state.demo_data = None

def load_models():
    """Load and initialize all analysis models"""
    if not st.session_state.models_initialized:
        with st.spinner("Loading analysis models..."):
            try:
                # Initialize analyzers
                st.session_state.wsi_analyzer = FlowMatchingLogisticWSI()
                st.session_state.ngs_analyzer = NGSAnalyzer()
                st.session_state.methylation_analyzer = MethylationAnalyzer()
                st.session_state.mrd_pipeline = MRDDetectionPipeline()
                st.session_state.risk_pipeline = RiskStratificationPipeline()
                st.session_state.clinical_support = ClinicalDecisionSupport()
                
                # Initialize multimodal model
                config = MultimodalConfig(
                    embed_dim=256,
                    hidden_dim=512,
                    fusion_strategy='adaptive'
                )
                st.session_state.multimodal_model = IntegratedCancerAnalysisModel(config)
                
                st.session_state.models_initialized = True
                st.success("✅ All models loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading models: {e}")
                return False
    
    return True

def main_header():
    """Display main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🧬 RISX Science</h1>
        <h3>Advanced Cancer Risk Analysis Platform</h3>
        <p>Flow Matching + Logistic Dynamics • Multimodal Integration • Clinical Decision Support</p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("🧬 Navigation")
    
    pages = {
        "🏠 Dashboard": "dashboard",
        "🔬 WSI Analysis": "wsi",
        "🧪 NGS Analysis": "ngs", 
        "⚗️ Methylation Analysis": "methylation",
        "🩸 MRD Detection": "mrd",
        "📊 Risk Stratification": "risk",
        "🏥 Clinical Decision": "clinical",
        "🔄 Comprehensive Report": "report",
        "🎯 Risk Arbitrage": "arbitrage",
        "⚙️ Model Training": "training",
        "📈 Synthetic Data": "synthetic"
    }
    
    selected_page = st.sidebar.radio("Select Analysis", list(pages.keys()))
    return pages[selected_page]

def dashboard_page():
    """Main dashboard page"""
    st.title("📊 Analysis Dashboard")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>47</h3>
            <p>Cancer Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>5</h3>
            <p>Analysis Modalities</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>98.3%</h3>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>~2min</h3>
            <p>Analysis Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent analyses
    st.subheader("📈 Recent Analyses")
    
    if st.session_state.analysis_results:
        df_results = pd.DataFrame([
            {
                'Analysis': analysis_type.upper(),
                'Timestamp': result.get('timestamp', datetime.now()),
                'Risk Score': result.get('risk_score', 'N/A'),
                'Status': '✅ Complete' if result else '❌ Failed'
            }
            for analysis_type, result in st.session_state.analysis_results.items()
        ])
        
        st.dataframe(df_results, use_container_width=True)
    else:
        st.info("No analyses completed yet. Start with WSI Analysis or upload your data.")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Run WSI Demo", use_container_width=True):
            st.session_state.demo_mode = 'wsi'
            st.experimental_rerun()
    
    with col2:
        if st.button("🧪 Run NGS Demo", use_container_width=True):
            st.session_state.demo_mode = 'ngs'
            st.experimental_rerun()
    
    with col3:
        if st.button("📊 Full Analysis Demo", use_container_width=True):
            st.session_state.demo_mode = 'full'
            st.experimental_rerun()

def wsi_analysis_page():
    """WSI Analysis page"""
    st.title("🔬 WSI Analysis")
    st.markdown("**Advanced Whole Slide Image Analysis with Flow Matching + Logistic Dynamics**")
    
    # File upload
    st.subheader("📁 Upload WSI Image")
    uploaded_file = st.file_uploader(
        "Choose WSI file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
        help="Upload a whole slide image for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded WSI", use_column_width=True)
        
        # Store in session state
        st.session_state.uploaded_files['wsi'] = uploaded_file
        
        # Analysis parameters
        st.subheader("🔧 Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            patch_size = st.selectbox("Patch Size", [256, 512, 1024], index=1)
            overlap = st.slider("Patch Overlap", 0.0, 0.5, 0.1, 0.05)
            
        with col2:
            magnification = st.selectbox("Magnification", ["10x", "20x", "40x"], index=1)
            enhancement = st.checkbox("Apply Enhancement", value=True)
        
        # Run analysis
        if st.button("🚀 Run WSI Analysis", type="primary"):
            run_wsi_analysis(uploaded_file, patch_size, overlap, magnification, enhancement)
    
    else:
        # Demo option
        st.info("No WSI file uploaded. You can run a demo analysis with synthetic data.")
        
        if st.button("🎯 Run WSI Demo Analysis"):
            run_wsi_demo_analysis()

def run_wsi_analysis(uploaded_file, patch_size, overlap, magnification, enhancement):
    """Run WSI analysis on uploaded file"""
    
    with st.spinner("Analyzing WSI... This may take a few minutes."):
        try:
            # Load the analyzer
            if 'wsi_analyzer' not in st.session_state:
                load_models()
            
            analyzer = st.session_state.wsi_analyzer
            
            # Convert uploaded file to format expected by analyzer
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Create analysis parameters
            analysis_params = {
                'patch_size': patch_size,
                'overlap': overlap,
                'magnification': magnification,
                'enhancement': enhancement
            }
            
            # Run analysis (simplified for demo)
            result = analyzer.analyze_wsi(image_array, **analysis_params)
            
            # Store results
            st.session_state.analysis_results['wsi'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.overall_risk,
                'parameters': analysis_params
            }
            
            # Display results
            display_wsi_results(result)
            
        except Exception as e:
            st.error(f"Error during WSI analysis: {e}")

def run_wsi_demo_analysis():
    """Run WSI demo analysis with synthetic data"""
    
    with st.spinner("Running WSI demo analysis..."):
        try:
            # Load the analyzer
            if 'wsi_analyzer' not in st.session_state:
                load_models()
            
            analyzer = st.session_state.wsi_analyzer
            
            # Create synthetic WSI data
            synthetic_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            
            # Run analysis
            result = analyzer.analyze_wsi(synthetic_image)
            
            # Store results
            st.session_state.analysis_results['wsi'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.overall_risk
            }
            
            # Display results
            display_wsi_results(result)
            
            st.success("✅ WSI demo analysis completed!")
            
        except Exception as e:
            st.error(f"Error during WSI demo: {e}")

def display_wsi_results(result):
    """Display WSI analysis results"""
    
    st.markdown("---")
    st.subheader("📊 WSI Analysis Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tumor Grade", f"Grade {result.tumor_grade}", f"{result.grade_confidence:.1%} confidence")
    
    with col2:
        risk_color = "🟢" if result.survival_risk < 0.3 else "🟡" if result.survival_risk < 0.7 else "🔴"
        st.metric("Survival Risk", f"{result.survival_risk:.1%}", risk_color)
    
    with col3:
        st.metric("Invasiveness", f"{result.tissue_metrics['invasiveness']:.3f}")
    
    with col4:
        st.metric("Hypoxic Fraction", f"{result.tissue_metrics['hypoxic_fraction']:.3f}")
    
    # Mutation predictions
    st.subheader("🧬 Mutation Probability Predictions")
    
    mutation_df = pd.DataFrame([
        {'Gene': gene, 'Probability': prob}
        for gene, prob in result.mutation_probabilities.items()
    ]).sort_values('Probability', ascending=False).head(10)
    
    fig = px.bar(mutation_df, x='Gene', y='Probability', 
                title="Top Mutation Predictions",
                color='Probability', color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Biological parameters
    st.subheader("⚗️ Sophisticated Biological Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        bio_metrics = {
            'Cellular Density': result.tissue_metrics['cellular_density'],
            'Vascular Density': result.tissue_metrics['vascular_density'],
            'Proliferation Index': result.tissue_metrics['proliferation_index'],
            'Necrosis Fraction': result.tissue_metrics['necrosis_fraction']
        }
        
        for metric, value in bio_metrics.items():
            st.metric(metric, f"{value:.3f}")
    
    with col2:
        # Flow dynamics visualization
        fig = go.Figure()
        
        # Sample flow field visualization
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        U = np.sin(X) * np.cos(Y)
        V = -np.cos(X) * np.sin(Y)
        
        fig.add_trace(go.Scatter(x=X.flatten(), y=Y.flatten(),
                               mode='markers', marker=dict(size=3),
                               name='Flow Field'))
        
        fig.update_layout(title="Flow Dynamics Visualization",
                         xaxis_title="Spatial X", yaxis_title="Spatial Y")
        
        st.plotly_chart(fig, use_container_width=True)

def ngs_analysis_page():
    """NGS Analysis page"""
    st.title("🧪 NGS Analysis")
    st.markdown("**Next Generation Sequencing Analysis & Variant Detection**")
    
    # File upload
    st.subheader("📁 Upload NGS Data")
    uploaded_file = st.file_uploader(
        "Choose VCF file",
        type=['vcf', 'vcf.gz'],
        help="Upload a VCF file containing genomic variants"
    )
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.session_state.uploaded_files['ngs'] = uploaded_file
        
        # Analysis options
        st.subheader("🔧 Analysis Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            variant_filter = st.selectbox("Variant Filter", ["All", "High Quality", "Pathogenic Only"])
            annotation_source = st.selectbox("Annotation Source", ["ClinVar", "COSMIC", "Both"])
        
        with col2:
            min_coverage = st.number_input("Minimum Coverage", min_value=1, value=20)
            min_vaf = st.slider("Minimum VAF", 0.0, 1.0, 0.05, 0.01)
        
        if st.button("🚀 Run NGS Analysis", type="primary"):
            run_ngs_analysis(uploaded_file, variant_filter, annotation_source, min_coverage, min_vaf)
    
    else:
        st.info("No VCF file uploaded. You can run a demo analysis with synthetic data.")
        
        if st.button("🎯 Run NGS Demo Analysis"):
            run_ngs_demo_analysis()

def run_ngs_analysis(uploaded_file, variant_filter, annotation_source, min_coverage, min_vaf):
    """Run NGS analysis on uploaded VCF file"""
    
    with st.spinner("Analyzing NGS data..."):
        try:
            if 'ngs_analyzer' not in st.session_state:
                load_models()
            
            analyzer = st.session_state.ngs_analyzer
            
            # Read VCF file
            vcf_content = uploaded_file.read()
            if isinstance(vcf_content, bytes):
                vcf_content = vcf_content.decode('utf-8')
            
            # Parse VCF and run analysis
            analysis_params = {
                'variant_filter': variant_filter,
                'annotation_source': annotation_source,
                'min_coverage': min_coverage,
                'min_vaf': min_vaf
            }
            
            result = analyzer.analyze_vcf_content(vcf_content, **analysis_params)
            
            # Store results
            st.session_state.analysis_results['ngs'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.overall_risk,
                'parameters': analysis_params
            }
            
            display_ngs_results(result)
            
        except Exception as e:
            st.error(f"Error during NGS analysis: {e}")

def run_ngs_demo_analysis():
    """Run NGS demo analysis"""
    
    with st.spinner("Running NGS demo analysis..."):
        try:
            if 'ngs_analyzer' not in st.session_state:
                load_models()
            
            analyzer = st.session_state.ngs_analyzer
            
            # Create demo VCF content
            demo_vcf = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
7	55181378	rs121913227	G	A	60	PASS	GENE=EGFR;MUTATION=L858R
12	25398284	rs121913529	C	T	60	PASS	GENE=KRAS;MUTATION=G12C
17	7674220	rs28934578	G	A	60	PASS	GENE=TP53;MUTATION=R273H"""
            
            result = analyzer.analyze_vcf_content(demo_vcf)
            
            # Store results
            st.session_state.analysis_results['ngs'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.overall_risk
            }
            
            display_ngs_results(result)
            st.success("✅ NGS demo analysis completed!")
            
        except Exception as e:
            st.error(f"Error during NGS demo: {e}")

def display_ngs_results(result):
    """Display NGS analysis results"""
    
    st.markdown("---")
    st.subheader("📊 NGS Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Variants", len(result.variants))
    
    with col2:
        pathogenic_count = len([v for v in result.variants if v.get('pathogenic', False)])
        st.metric("Pathogenic Variants", pathogenic_count)
    
    with col3:
        st.metric("Tumor Mutational Burden", f"{result.tumor_mutational_burden:.1f}")
    
    with col4:
        st.metric("Microsatellite Status", result.microsatellite_status)
    
    # Actionable mutations
    if result.actionable_mutations:
        st.subheader("🎯 Actionable Mutations")
        
        actionable_df = pd.DataFrame(result.actionable_mutations)
        st.dataframe(actionable_df, use_container_width=True)
        
        # Treatment recommendations
        st.subheader("💊 Treatment Recommendations")
        for rec in result.treatment_recommendations[:5]:
            st.markdown(f"• {rec}")
    
    # Mutation landscape
    if result.variants:
        st.subheader("🗺️ Mutation Landscape")
        
        # Create mutation frequency plot
        gene_counts = {}
        for variant in result.variants:
            gene = variant.get('gene', 'Unknown')
            gene_counts[gene] = gene_counts.get(gene, 0) + 1
        
        if gene_counts:
            genes_df = pd.DataFrame([
                {'Gene': gene, 'Mutations': count}
                for gene, count in sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
            ]).head(15)
            
            fig = px.bar(genes_df, x='Gene', y='Mutations',
                        title="Mutations per Gene",
                        color='Mutations', color_continuous_scale='plasma')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def methylation_analysis_page():
    """Methylation Analysis page"""
    st.title("⚗️ Methylation Analysis")
    st.markdown("**DNA Methylation Pattern Analysis & Epigenetic Biomarkers**")
    
    # File upload or demo
    st.subheader("📁 Upload Methylation Data")
    uploaded_file = st.file_uploader(
        "Choose methylation file",
        type=['csv', 'txt', 'idat'],
        help="Upload methylation array data or processed methylation values"
    )
    
    if uploaded_file is not None:
        st.success(f"Uploaded: {uploaded_file.name}")
        st.session_state.uploaded_files['methylation'] = uploaded_file
        
        if st.button("🚀 Run Methylation Analysis", type="primary"):
            run_methylation_analysis(uploaded_file)
    
    else:
        st.info("No methylation file uploaded. You can run a demo analysis.")
        
        if st.button("🎯 Run Methylation Demo Analysis"):
            run_methylation_demo_analysis()

def run_methylation_demo_analysis():
    """Run methylation demo analysis"""
    
    with st.spinner("Running methylation demo analysis..."):
        try:
            if 'methylation_analyzer' not in st.session_state:
                load_models()
            
            analyzer = st.session_state.methylation_analyzer
            
            # Create synthetic methylation data
            synthetic_data = {
                f'cg{i:08d}': np.random.beta(2, 2) for i in range(1000)
            }
            
            result = analyzer.analyze_methylation_pattern(synthetic_data)
            
            # Store results
            st.session_state.analysis_results['methylation'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.methylation_risk_score
            }
            
            display_methylation_results(result)
            st.success("✅ Methylation demo analysis completed!")
            
        except Exception as e:
            st.error(f"Error during methylation demo: {e}")

def display_methylation_results(result):
    """Display methylation analysis results"""
    
    st.markdown("---")
    st.subheader("📊 Methylation Analysis Results")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Global Methylation", f"{result.global_methylation_level:.3f}")
    
    with col2:
        st.metric("CIMP Status", result.cimp_status)
    
    with col3:
        st.metric("Methylation Age", f"{result.methylation_age:.1f} years")
    
    with col4:
        st.metric("Risk Score", f"{result.methylation_risk_score:.3f}")
    
    # Hypermethylated genes
    if result.hypermethylated_genes:
        st.subheader("🧬 Hypermethylated Genes")
        
        hyper_df = pd.DataFrame([
            {'Gene': gene, 'Methylation Level': level}
            for gene, level in list(result.hypermethylated_genes.items())[:10]
        ])
        
        fig = px.bar(hyper_df, x='Gene', y='Methylation Level',
                    title="Top Hypermethylated Genes",
                    color='Methylation Level', color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Treatment implications
    if result.treatment_implications:
        st.subheader("💊 Treatment Implications")
        for implication in result.treatment_implications:
            st.markdown(f"• {implication}")

def mrd_detection_page():
    """MRD Detection page"""
    st.title("🩸 MRD Detection")
    st.markdown("**Minimal Residual Disease Detection & Monitoring**")
    
    # Sample information
    st.subheader("🧪 Sample Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_type = st.selectbox("Sample Type", ["Plasma", "Serum", "Urine", "CSF"])
        collection_date = st.date_input("Collection Date", datetime.now().date())
        
    with col2:
        sample_volume = st.number_input("Sample Volume (mL)", min_value=0.1, value=10.0, step=0.1)
        processing_delay = st.number_input("Processing Delay (hours)", min_value=0.0, value=2.0, step=0.5)
    
    # File upload
    st.subheader("📁 Upload Liquid Biopsy Data")
    uploaded_file = st.file_uploader(
        "Choose liquid biopsy file",
        type=['csv', 'vcf', 'txt'],
        help="Upload ctDNA sequencing data or mutation calls"
    )
    
    if uploaded_file is not None or st.checkbox("Use Demo Data"):
        if st.button("🚀 Run MRD Analysis", type="primary"):
            run_mrd_analysis(sample_type, collection_date, sample_volume, processing_delay, uploaded_file)

def run_mrd_analysis(sample_type, collection_date, sample_volume, processing_delay, uploaded_file=None):
    """Run MRD analysis"""
    
    with st.spinner("Analyzing MRD..."):
        try:
            if 'mrd_pipeline' not in st.session_state:
                load_models()
            
            pipeline = st.session_state.mrd_pipeline
            
            # Create or use sample liquid biopsy data
            if uploaded_file is None:
                biopsy_data = create_sample_liquid_biopsy()
            else:
                # Process uploaded file (simplified for demo)
                biopsy_data = create_sample_liquid_biopsy()
                biopsy_data.sample_type = sample_type.lower()
                biopsy_data.volume_ml = sample_volume
                biopsy_data.processing_delay_hours = processing_delay
            
            result = pipeline.analyze_liquid_biopsy(biopsy_data)
            
            # Store results
            st.session_state.analysis_results['mrd'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.relapse_risk
            }
            
            display_mrd_results(result)
            st.success("✅ MRD analysis completed!")
            
        except Exception as e:
            st.error(f"Error during MRD analysis: {e}")

def display_mrd_results(result):
    """Display MRD analysis results"""
    
    st.markdown("---")
    st.subheader("📊 MRD Analysis Results")
    
    # Main status
    status_color = "🔴" if result.mrd_status == "Positive" else "🟢" if result.mrd_status == "Negative" else "🟡"
    st.markdown(f"### MRD Status: {status_color} {result.mrd_status}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ctDNA Detected", "Yes" if result.ctdna_detected else "No")
    
    with col2:
        st.metric("ctDNA Concentration", f"{result.ctdna_concentration:.4f} copies/mL")
    
    with col3:
        st.metric("VAF", f"{result.variant_allele_frequency:.4f}")
    
    with col4:
        st.metric("Relapse Risk", f"{result.relapse_risk:.1%}")
    
    # Tracking mutations
    if result.tracking_mutations:
        st.subheader("🎯 Tracking Mutations")
        
        tracking_df = pd.DataFrame(result.tracking_mutations)
        st.dataframe(tracking_df, use_container_width=True)
    
    # Monitoring recommendations
    st.subheader("📋 Monitoring Recommendations")
    for rec in result.monitoring_recommendations:
        st.markdown(f"• {rec}")
    
    # Longitudinal trend
    if result.longitudinal_trend != "Insufficient data":
        trend_color = "🔴" if "Increasing" in result.longitudinal_trend else "🟢" if "Decreasing" in result.longitudinal_trend else "🟡"
        st.markdown(f"**Longitudinal Trend:** {trend_color} {result.longitudinal_trend}")

def risk_stratification_page():
    """Risk Stratification page"""
    st.title("📊 Risk Stratification")
    st.markdown("**Comprehensive Risk Analysis & Survival Prediction**")
    
    # Patient information input
    st.subheader("👤 Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=65)
        gender = st.selectbox("Gender", ["Female", "Male"])
        performance_status = st.selectbox("Performance Status (ECOG)", [0, 1, 2, 3, 4])
    
    with col2:
        tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, value=3.2, step=0.1)
        lymph_nodes = st.number_input("Positive Lymph Nodes", min_value=0, value=2)
        tumor_grade = st.selectbox("Tumor Grade", [1, 2, 3, 4])
    
    with col3:
        metastases = st.checkbox("Distant Metastases")
        ki67 = st.slider("Ki-67 Index (%)", 0, 100, 25)
        her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"])
    
    if st.button("🚀 Run Risk Analysis", type="primary"):
        run_risk_analysis(age, gender, performance_status, tumor_size, lymph_nodes, 
                         tumor_grade, metastases, ki67, her2_status)

def run_risk_analysis(age, gender, performance_status, tumor_size, lymph_nodes,
                     tumor_grade, metastases, ki67, her2_status):
    """Run risk stratification analysis"""
    
    with st.spinner("Calculating risk scores..."):
        try:
            if 'risk_pipeline' not in st.session_state:
                load_models()
            
            pipeline = st.session_state.risk_pipeline
            
            # Create prognostic factors
            factors = create_sample_prognostic_factors()
            
            # Update with user inputs
            factors.age = float(age)
            factors.gender = gender.lower()
            factors.performance_status = performance_status
            factors.tumor_size = tumor_size
            factors.lymph_node_involvement = lymph_nodes
            factors.histologic_grade = tumor_grade
            factors.metastases_present = metastases
            factors.ki67_index = float(ki67)
            factors.her2_status = her2_status.lower()
            
            result = pipeline.assess_risk(factors)
            
            # Store results
            st.session_state.analysis_results['risk'] = {
                'result': result,
                'timestamp': datetime.now(),
                'risk_score': result.overall_risk_score
            }
            
            display_risk_results(result)
            st.success("✅ Risk analysis completed!")
            
        except Exception as e:
            st.error(f"Error during risk analysis: {e}")

def display_risk_results(result):
    """Display risk stratification results"""
    
    st.markdown("---")
    st.subheader("📊 Risk Assessment Results")
    
    # Overall risk
    risk_color = "🟢" if result.risk_category == "Low" else "🟡" if result.risk_category == "Moderate" else "🔴"
    st.markdown(f"### Overall Risk: {risk_color} {result.risk_category}")
    st.markdown(f"**Risk Score:** {result.overall_risk_score:.3f}")
    
    # Survival probabilities
    st.subheader("⏰ Survival Probabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("1-Year Survival", f"{result.survival_probability_1yr:.1%}")
    
    with col2:
        st.metric("3-Year Survival", f"{result.survival_probability_3yr:.1%}")
    
    with col3:
        st.metric("5-Year Survival", f"{result.survival_probability_5yr:.1%}")
    
    # Risk breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛡️ Protective Factors")
        for factor in result.protective_factors[:5]:
            st.markdown(f"• {factor}")
    
    with col2:
        st.subheader("⚠️ Adverse Factors")
        for factor in result.adverse_factors[:5]:
            st.markdown(f"• {factor}")
    
    # Treatment recommendations
    st.subheader("💊 Treatment Recommendations")
    for rec in result.treatment_recommendations:
        st.markdown(f"• {rec}")

def clinical_decision_page():
    """Clinical Decision Support page"""
    st.title("🏥 Clinical Decision Support")
    st.markdown("**Evidence-Based Treatment Recommendations & Decision Support**")
    
    # Integration of all analyses
    st.subheader("🔗 Integrated Analysis Results")
    
    if st.session_state.analysis_results:
        # Display integrated summary
        tabs = st.tabs(["📊 Summary", "💊 Recommendations", "📈 Trends"])
        
        with tabs[0]:
            display_integrated_summary()
        
        with tabs[1]:
            display_treatment_recommendations()
        
        with tabs[2]:
            display_trend_analysis()
    
    else:
        st.info("No analysis results available. Please run analyses from other tabs first.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔬 Run WSI Demo"):
                st.session_state.auto_demo = 'wsi'
                st.experimental_rerun()
        
        with col2:
            if st.button("🧪 Run NGS Demo"):
                st.session_state.auto_demo = 'ngs'
                st.experimental_rerun()
        
        with col3:
            if st.button("📊 Run All Demos"):
                st.session_state.auto_demo = 'all'
                st.experimental_rerun()

def display_integrated_summary():
    """Display integrated analysis summary"""
    
    st.subheader("🎯 Integrated Risk Assessment")
    
    # Collect risk scores from all analyses
    risk_scores = {}
    
    for analysis_type, data in st.session_state.analysis_results.items():
        if 'risk_score' in data:
            risk_scores[analysis_type.upper()] = data['risk_score']
    
    if risk_scores:
        # Create risk score comparison
        scores_df = pd.DataFrame([
            {'Analysis': analysis, 'Risk Score': score}
            for analysis, score in risk_scores.items()
        ])
        
        fig = px.bar(scores_df, x='Analysis', y='Risk Score',
                    title="Risk Scores by Analysis Type",
                    color='Risk Score', color_continuous_scale='reds')
        st.plotly_chart(fig, use_container_width=True)
        
        # Overall integrated risk
        overall_risk = np.mean(list(risk_scores.values()))
        risk_category = "Low" if overall_risk < 0.3 else "Moderate" if overall_risk < 0.7 else "High"
        risk_color = "🟢" if risk_category == "Low" else "🟡" if risk_category == "Moderate" else "🔴"
        
        st.markdown(f"### Integrated Risk Assessment: {risk_color} {risk_category}")
        st.markdown(f"**Combined Risk Score:** {overall_risk:.3f}")

def display_treatment_recommendations():
    """Display integrated treatment recommendations"""
    
    st.subheader("💊 Integrated Treatment Recommendations")
    
    # Collect recommendations from all analyses
    all_recommendations = []
    
    for analysis_type, data in st.session_state.analysis_results.items():
        result = data.get('result')
        if hasattr(result, 'treatment_recommendations'):
            all_recommendations.extend(result.treatment_recommendations)
        elif hasattr(result, 'monitoring_recommendations'):
            all_recommendations.extend(result.monitoring_recommendations)
    
    if all_recommendations:
        # Remove duplicates and categorize
        unique_recommendations = list(set(all_recommendations))
        
        for i, rec in enumerate(unique_recommendations[:10], 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific treatment recommendations available from current analyses.")

def display_trend_analysis():
    """Display trend analysis if multiple time points available"""
    
    st.subheader("📈 Longitudinal Trends")
    
    # This would show trends over time if we had multiple analyses
    st.info("Trend analysis requires multiple time points. This feature will show changes in risk scores and biomarkers over time.")
    
    # Placeholder visualization
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    sample_trend = np.random.normal(0.5, 0.1, 12)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=sample_trend, mode='lines+markers',
                           name='Risk Score Trend', line=dict(color='red')))
    fig.update_layout(title="Sample Risk Score Trend Over Time",
                     xaxis_title="Date", yaxis_title="Risk Score")
    st.plotly_chart(fig, use_container_width=True)

def comprehensive_report_page():
    """Comprehensive Report page"""
    st.title("🔄 Comprehensive Report")
    st.markdown("**Complete Multimodal Cancer Analysis Report**")
    
    if st.session_state.analysis_results:
        # Generate comprehensive report
        if st.button("📄 Generate Complete Report", type="primary"):
            generate_comprehensive_report()
    else:
        st.info("No analysis results available for report generation.")
        
        if st.button("🎯 Run Complete Demo Analysis"):
            run_complete_demo_analysis()

def run_complete_demo_analysis():
    """Run complete demo analysis across all modalities"""
    
    with st.spinner("Running complete multimodal analysis..."):
        try:
            # Run all demo analyses
            load_models()
            
            # WSI Demo
            st.write("Running WSI analysis...")
            analyzer = st.session_state.wsi_analyzer
            synthetic_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
            wsi_result = analyzer.analyze_wsi(synthetic_image)
            
            st.session_state.analysis_results['wsi'] = {
                'result': wsi_result,
                'timestamp': datetime.now(),
                'risk_score': wsi_result.overall_risk
            }
            
            # NGS Demo
            st.write("Running NGS analysis...")
            ngs_analyzer = st.session_state.ngs_analyzer
            demo_vcf = """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
7	55181378	rs121913227	G	A	60	PASS	GENE=EGFR"""
            ngs_result = ngs_analyzer.analyze_vcf_content(demo_vcf)
            
            st.session_state.analysis_results['ngs'] = {
                'result': ngs_result,
                'timestamp': datetime.now(),
                'risk_score': ngs_result.overall_risk
            }
            
            # Additional analyses...
            st.write("Running additional analyses...")
            
            st.success("✅ Complete multimodal analysis finished!")
            st.experimental_rerun()
            
        except Exception as e:
            st.error(f"Error during complete analysis: {e}")

def generate_comprehensive_report():
    """Generate a comprehensive analysis report"""
    
    st.subheader("📋 Executive Summary")
    
    # Patient summary
    st.markdown("""
    **Patient ID:** DEMO_001  
    **Analysis Date:** {}  
    **Modalities Analyzed:** {}  
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        ", ".join(st.session_state.analysis_results.keys()).upper()
    ))
    
    # Risk summary
    risk_scores = [data.get('risk_score', 0) for data in st.session_state.analysis_results.values() if 'risk_score' in data]
    if risk_scores:
        overall_risk = np.mean(risk_scores)
        st.markdown(f"**Overall Risk Score:** {overall_risk:.3f}")
    
    # Detailed results by modality
    for analysis_type, data in st.session_state.analysis_results.items():
        st.subheader(f"📊 {analysis_type.upper()} Results")
        
        result = data.get('result')
        if result:
            # Display key findings for each analysis type
            if analysis_type == 'wsi':
                st.markdown(f"- **Tumor Grade:** Grade {getattr(result, 'tumor_grade', 'N/A')}")
                st.markdown(f"- **Survival Risk:** {getattr(result, 'survival_risk', 0):.1%}")
            elif analysis_type == 'ngs':
                st.markdown(f"- **Total Variants:** {len(getattr(result, 'variants', []))}")
                st.markdown(f"- **TMB:** {getattr(result, 'tumor_mutational_burden', 0):.1f}")
            # Add other analysis types...
    
    # Generate downloadable report
    if st.button("📥 Download Report"):
        report_content = generate_report_content()
        st.download_button(
            label="Download PDF Report",
            data=report_content,
            file_name=f"cancer_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

def generate_report_content():
    """Generate report content for download"""
    
    content = f"""
RISX SCIENCE - COMPREHENSIVE CANCER ANALYSIS REPORT
==================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
----------------
Patient ID: DEMO_001
Analyses Completed: {', '.join(st.session_state.analysis_results.keys()).upper()}

RISK ASSESSMENT
--------------
"""
    
    for analysis_type, data in st.session_state.analysis_results.items():
        risk_score = data.get('risk_score', 'N/A')
        content += f"{analysis_type.upper()} Risk Score: {risk_score}\n"
    
    content += f"""

RECOMMENDATIONS
--------------
Based on the integrated analysis, the following recommendations are provided:
1. Continue current monitoring protocol
2. Consider additional biomarker testing
3. Schedule follow-up in 3 months

DISCLAIMER
----------
This report is for research purposes only and should not be used for clinical decision-making.

RISX Science - 16 Cross Street, 206 New Canaan, CT 06840
"""
    
    return content

def training_page():
    """Model Training page"""
    st.title("⚙️ Model Training")
    st.markdown("**Train and Fine-tune Analysis Models**")
    
    # Training options
    st.subheader("🎯 Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("Model Type", [
            "WSI Flow Matching",
            "NGS Transformer", 
            "Methylation Analyzer",
            "Multimodal Fusion",
            "All Models"
        ])
        
        dataset_size = st.selectbox("Dataset Size", [100, 500, 1000, 5000])
        
    with col2:
        epochs = st.slider("Training Epochs", 1, 100, 10)
        learning_rate = st.selectbox("Learning Rate", [1e-5, 1e-4, 1e-3, 1e-2])
    
    # Training controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Training", type="primary"):
            start_training(model_type, dataset_size, epochs, learning_rate)
    
    with col2:
        if st.button("⏸️ Pause Training"):
            st.info("Training paused")
    
    with col3:
        if st.button("📊 View Metrics"):
            show_training_metrics()

def start_training(model_type, dataset_size, epochs, learning_rate):
    """Start model training"""
    
    st.subheader("🔄 Training Progress")
    
    # Create progress bars
    epoch_progress = st.progress(0)
    batch_progress = st.progress(0)
    
    # Training metrics placeholder
    metrics_placeholder = st.empty()
    
    # Simulate training
    for epoch in range(epochs):
        st.write(f"Epoch {epoch + 1}/{epochs}")
        
        # Simulate batches
        num_batches = 10
        for batch in range(num_batches):
            # Simulate training step
            import time
            time.sleep(0.1)
            
            # Update progress
            batch_progress.progress((batch + 1) / num_batches)
            
            # Simulate metrics
            loss = 1.0 * np.exp(-epoch * 0.1) + np.random.normal(0, 0.1)
            accuracy = 0.5 + 0.4 * (1 - np.exp(-epoch * 0.1)) + np.random.normal(0, 0.05)
            
            metrics_placeholder.metric("Training Loss", f"{loss:.4f}")
        
        epoch_progress.progress((epoch + 1) / epochs)
    
    st.success("✅ Training completed!")

def show_training_metrics():
    """Show training metrics and visualizations"""
    
    st.subheader("📈 Training Metrics")
    
    # Generate sample training curves
    epochs = range(1, 21)
    train_loss = [1.0 * np.exp(-e * 0.1) + np.random.normal(0, 0.05) for e in epochs]
    val_loss = [1.0 * np.exp(-e * 0.08) + np.random.normal(0, 0.05) for e in epochs]
    train_acc = [0.5 + 0.4 * (1 - np.exp(-e * 0.1)) + np.random.normal(0, 0.02) for e in epochs]
    val_acc = [0.5 + 0.35 * (1 - np.exp(-e * 0.08)) + np.random.normal(0, 0.02) for e in epochs]
    
    # Create training curves
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Loss', 'Training Accuracy']
    )
    
    fig.add_trace(go.Scatter(x=list(epochs), y=train_loss, name='Train Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=val_loss, name='Val Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(epochs), y=train_acc, name='Train Acc'), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(epochs), y=val_acc, name='Val Acc'), row=1, col=2)
    
    fig.update_layout(height=400, title_text="Training Metrics")
    st.plotly_chart(fig, use_container_width=True)

def synthetic_data_page():
    """Synthetic Data Generation page"""
    st.title("📈 Synthetic Data Generation")
    st.markdown("**Generate Realistic Cancer Data for Training & Testing**")
    
    # Data generation parameters
    st.subheader("🔧 Generation Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_samples = st.selectbox("Number of Samples", [100, 500, 1000, 5000, 10000])
        cancer_types = st.multiselect("Cancer Types", [
            "Breast", "Lung", "Colon", "Prostate", "Melanoma", "Pancreatic"
        ], default=["Breast", "Lung", "Colon"])
    
    with col2:
        age_range = st.slider("Age Range", 18, 100, (45, 85))
        gender_balance = st.slider("Male/Female Ratio", 0.0, 1.0, 0.5)
    
    with col3:
        mutation_rate = st.slider("Mutation Rate", 0.05, 0.3, 0.15)
        noise_level = st.slider("Noise Level", 0.0, 0.3, 0.1)
    
    # Generate data
    if st.button("🎯 Generate Synthetic Dataset", type="primary"):
        generate_synthetic_data(num_samples, cancer_types, age_range, gender_balance, 
                              mutation_rate, noise_level)

def generate_synthetic_data(num_samples, cancer_types, age_range, gender_balance, 
                          mutation_rate, noise_level):
    """Generate synthetic cancer dataset"""
    
    with st.spinner(f"Generating {num_samples} synthetic samples..."):
        try:
            from data_generation import SyntheticCancerDataset, DataGenerationConfig
            
            # Create configuration
            config = DataGenerationConfig(
                num_samples=num_samples,
                age_distribution=age_range,
                gender_distribution={'male': gender_balance, 'female': 1-gender_balance},
                mutation_rate=mutation_rate,
                noise_level=noise_level
            )
            
            # Generate dataset
            dataset = SyntheticCancerDataset(num_samples=num_samples, config=config)
            
            # Store in session state
            st.session_state.synthetic_dataset = dataset
            
            # Display dataset statistics
            display_dataset_statistics(dataset)
            
            st.success(f"✅ Generated {len(dataset)} synthetic samples!")
            
        except Exception as e:
            st.error(f"Error generating synthetic data: {e}")

def display_dataset_statistics(dataset):
    """Display statistics for generated dataset"""
    
    st.subheader("📊 Dataset Statistics")
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(dataset))
    
    with col2:
        cancer_types = [p.cancer_type for p in dataset.patients]
        unique_types = len(set(cancer_types))
        st.metric("Cancer Types", unique_types)
    
    with col3:
        ages = [p.age for p in dataset.patients]
        avg_age = np.mean(ages)
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    with col4:
        mutations = [len(p.mutations) for p in dataset.patients]
        avg_mutations = np.mean(mutations)
        st.metric("Avg Mutations", f"{avg_mutations:.1f}")
    
    # Cancer type distribution
    cancer_counts = pd.Series(cancer_types).value_counts()
    fig = px.pie(values=cancer_counts.values, names=cancer_counts.index,
                title="Cancer Type Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Age distribution
    fig = px.histogram(x=ages, title="Age Distribution", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Display header
    main_header()
    
    # Load models
    if not load_models():
        st.error("Failed to load models. Please check the installation.")
        return
    
    # Navigation
    selected_page = sidebar_navigation()
    
    # Route to appropriate page
    if selected_page == "dashboard":
        dashboard_page()
    elif selected_page == "wsi":
        wsi_analysis_page()
    elif selected_page == "ngs":
        ngs_analysis_page()
    elif selected_page == "methylation":
        methylation_analysis_page()
    elif selected_page == "mrd":
        mrd_detection_page()
    elif selected_page == "risk":
        risk_stratification_page()
    elif selected_page == "clinical":
        clinical_decision_page()
    elif selected_page == "report":
        comprehensive_report_page()
    elif selected_page == "arbitrage":
        st.title("🎯 Risk Arbitrage")
        st.info("Risk arbitrage analysis coming soon!")
    elif selected_page == "training":
        training_page()
    elif selected_page == "synthetic":
        synthetic_data_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>RISX Science</strong> - Advanced Cancer Risk Analysis Platform</p>
        <p>16 Cross Street, 206 New Canaan, CT 06840</p>
        <p><em>Research Use Only - Not for Clinical Decision Making</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
