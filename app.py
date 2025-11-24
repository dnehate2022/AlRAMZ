import streamlit as st
import pandas as pd
import json
import os
import base64
from datetime import datetime
import tempfile
import fitz  # PyMuPDF

# Don't import backend at module level - delay until needed
# This prevents DSPy reconfiguration on every Streamlit rerun

# Page config
st.set_page_config(page_title="Trading Order Extractor", page_icon="📄", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stColumn {
        padding: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] > div {
        border-radius: 8px;
        padding: 1rem;
    }
.st-emotion-cache-zy6yx3 {
    width: 100%;
    padding: 3rem 1rem 3rem;
    max-width: initial;
    min-width: auto;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'extraction_result' not in st.session_state:
    st.session_state.extraction_result = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'token_usage' not in st.session_state:
    st.session_state.token_usage = None
if 'dspy_configured' not in st.session_state:
    st.session_state.dspy_configured = False

def display_pdf_as_images(pdf_path):
    """Display PDF pages as images"""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        st.image(img_data, caption=f"Page {page_num + 1}")
    doc.close()

def display_pdf_embedded(pdf_path):
    """Display PDF using iframe"""
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def result_to_dataframe(result):
    """Convert extraction result to DataFrame"""
    exclude_keys = ['combined_input_token', 'combined_output_token', 'combined_total_token']
    data = {'Field': [], 'Value': []}
    
    for key, value in result.items():
        if key not in exclude_keys:
            field_name = key.replace('_', ' ').title()
            data['Field'].append(field_name)
            data['Value'].append(str(value) if value else "")
    
    return pd.DataFrame(data)

def dataframe_to_dict(df):
    """Convert DataFrame back to dict"""
    result = {}
    for _, row in df.iterrows():
        key = row['Field'].lower().replace(' ', '_')
        result[key] = row['Value']
    return result

# Main UI
st.markdown('<div class="main-header">📄 Trading Order Extractor</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    force_retrain = st.checkbox("Force Model Retraining", value=False)
    
    # Model cache status
    model_exists = os.path.exists("<models/trading_extractor.json")
    if model_exists:
        st.success("✅ Trained model cached")
    else:
        st.info("ℹ️ No cached model (will train on first run)")
    
    st.markdown("---")
    # st.markdown("### 🤖 Models")
    # st.info("**OCR:** Gemini 2.0 Flash\n\n**Extract:** Gemini 2.5 Flash")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 1️⃣ Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.uploaded_file_path = tmp.name
        st.success(f"✅ Uploaded: {uploaded_file.name}")

with col2:
    st.markdown("### 2️⃣ Extract Data")
    if st.session_state.uploaded_file_path:
        if st.button("🚀 Extract Information", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Import backend only when needed (not at module level)
                    # This prevents DSPy reconfiguration on every Streamlit rerun
                    from main import main_trading_extraction
                    
                    # Show model status
                    model_path = "models/trading_extractor.json"
                    if os.path.exists(model_path) and not force_retrain:
                        st.info("ℹ️ Using cached trained model (faster)...")
                    else:
                        st.warning("⚠️ Training new model (first time or forced retrain)...")
                    
                    # Training example data
                    example_data = {
                        'example_1': {
                            "pdf_path": "FirstOrderSlip.pdf",
                            "broker": "Al Ramz",
                            "market": "السوق (ADX)",
                            "date": "4/7/21",
                            "time": "9:32",
                            "action_type": "شراء (Buy)",
                            "investor_name": "Mim Okta Mininnic Co.",
                            "guardian_attorney_name": "محمد عبد المالك",
                            "trading_account_Type": "نقدي (Cash)",
                            "trading_account_number": "105",
                            "security_name": "Borouge",
                            "security_volume_quantity": "50000",
                            "security_volume_quantity_unit": "عدد الأوراق (Quantity)",
                            "order_type": "سعر محدد (Limit Price)",
                            "order_type_price": "2.6",
                            "order_validity": "يومي (Daily)",
                            "authorized_signatory_name": "Rahal Kamarji",
                            "authorized_signatory_code": "ARC-218"
                        }
                    }
                    
                    # Call your existing backend function
                    result, input_tokens, output_tokens, total_tokens = main_trading_extraction(
                        pdf_path=st.session_state.uploaded_file_path,
                        trading_examples=example_data,
                        force_retrain=force_retrain
                    )
                    
                    if result:
                        st.session_state.extraction_result = result
                        st.session_state.token_usage = {
                            'input': input_tokens,
                            'output': output_tokens,
                            'total': total_tokens
                        }
                        st.success("✅ Extraction completed!")
                        st.rerun()
                    else:
                        st.error("❌ Extraction failed")
                        
                except RuntimeError as e:
                    if "dspy.settings" in str(e):
                        st.error("⚠️ DSPy Configuration Error Detected")
                        st.warning("""
                        **What happened:** DSPy can only be configured once per session.  
                        **Solution:** Restart the Streamlit app.
                        """)
                        st.code("# Stop the app (Ctrl+C) and restart:\nstreamlit run app.py", language="bash")
                    else:
                        st.error(f"❌ Runtime Error: {str(e)}")
                        with st.expander("🔍 Error Details"):
                            st.exception(e)
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")
                    with st.expander("🔍 Full Error Traceback"):
                        st.exception(e)
                    st.exception(e)
    else:
        st.info("👆 Upload a PDF first")

# Results
if st.session_state.extraction_result:
    st.markdown("---")
    st.markdown("### 📊 Extraction Results")
    
    # Token usage
    if st.session_state.token_usage:
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Tokens", st.session_state.token_usage['input'])
        col2.metric("Output Tokens", st.session_state.token_usage['output'])
        col3.metric("Total Tokens", st.session_state.token_usage['total'])
    
    st.markdown("---")
    
    # Main layout: PDF Viewer (left) and Save/Export (right) in 6-6 columns
    col_left, col_right = st.columns([7, 5])
    
    # Left column: PDF Viewer
    with col_left:
        st.markdown("#### 📖 PDF Viewer")
        view_mode = st.radio("View Mode:", ["Image Preview", "Embedded PDF"], horizontal=True, key="results_view")
        
        if view_mode == "Image Preview":
            display_pdf_as_images(st.session_state.uploaded_file_path)
        else:
            display_pdf_embedded(st.session_state.uploaded_file_path)
    
    # Right column: JSON View + Download
    with col_right:
        st.markdown("#### 🔍 Extracted Data (JSON)")
        
        # Show JSON directly
        st.json(st.session_state.extraction_result)
        
        # Download buttons
        st.markdown("#### 📥 Download")
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            # Prepare JSON data for download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_data = st.session_state.extraction_result.copy()
            json_filename = f"trading_order_{timestamp}.json"
            
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(json_data, indent=2, ensure_ascii=False),
                file_name=json_filename,
                mime="application/json"
            )
        
        with dl_col2:
            # Prepare CSV data
            df = result_to_dataframe(st.session_state.extraction_result)
            csv_filename = f"trading_order_{timestamp}.csv"
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv"
            )
        
        with dl_col3:
            if st.button("🔄 Reset"):
                st.session_state.extraction_result = None
                st.session_state.token_usage = None
                st.session_state.uploaded_file_path = None
                st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #7f8c8d;'>Powered by DSPy + Gemini + Streamlit</div>", unsafe_allow_html=True)