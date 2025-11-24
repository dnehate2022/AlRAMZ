import json
import os
import sys
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from urllib.parse import unquote, urlparse
import posixpath
import streamlit as st
import fitz
from PIL import Image
import dspy
from pydantic import BaseModel
import requests
import google.generativeai as genai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration from environment variables"""
    GEMINI_API_KEY = st.secrets.get("GEMINI_FLASH_API_KEY", os.getenv("GEMINI_FLASH_API_KEY"))
    MODELS_DIR = st.secrets.get("MODELS_DIR", os.getenv("MODELS_DIR", "./models"))

# ============================================================================
# Pydantic Model - Flattened Structure
# ============================================================================

class TradingOrderInfo(BaseModel):
    """Trading order form information - flattened structure"""
    broker: str
    market: str
    date: str
    time: str
    action_type: str
    investor_name: str
    guardian_attorney_name: str
    trading_account_Type: str
    trading_account_number: str
    security_name: str
    security_volume_quantity: str
    security_volume_quantity_unit: str
    order_type: str
    order_type_price: str
    order_validity: str
    authorized_signatory_name: str
    authorized_signatory_code: str

# ============================================================================
# DSPy Signature
# ============================================================================

class TradingOrderExtractor(dspy.Signature):
    """Extract trading order information from Al Ramz and similar trading forms.
    
    Extract information from trading order documents using BOTH images and text.
    
    CRITICAL INSTRUCTIONS:
    - Use the extracted text along with images for accurate data extraction
    - Extract EXACT values as they appear in the document
    - Support both Arabic and English text
    - Handle bilingual forms (Arabic/English)
    - Preserve both Arabic and English labels in format: "Arabic (English)"
    
    IMPORTANT: Reference the provided training examples to understand the extraction patterns.
    Follow the same logic used in the training examples for similar document types.
    
    - Please disregard any previous conversation context and treat the following text as a completely new, standalone document.
    - Clear previous instruction memory
    
    CRITICAL - FIELD EXTRACTION GUIDELINES:
    
    broker = Company/Broker Name
    - Look for: Logo and company name at top (e.g., "Al Ramz", "الرمز")
    - Extract exact name as displayed
    
    market = Trading Market/Exchange
    - Look for: "Market:", "السوق", "ADX", "DFM", "NASDAQ", etc.
    - Format: "Arabic (English)" if both present, e.g., "السوق (ADX)"
    - if 'AX' THEN Convert to "ADX"
    
    date = Transaction Date
    - Look for: "Date:", "التاريخ", date fields in header
    - Format: MM/DD/YY or DD/MM/YYYY as shown in document
    
    time = Transaction Time
    - Look for: "Time:", "الوقت", timestamp in header
    - Format: HH:MM (24-hour format preferred)
    - If only 12-hour format available, keep as is (e.g., "9:32")
    
    action_type = Order Action Type
    - Look for: "Action Type:", "نوع العملية"
    - Checkboxes/options: شراء (Buy), بيع (Sell), تعديل (Modify), إلغاء (Cancel)
    - Extract the SELECTED option with both languages
    
    investor_name = Investor/Client Name
    - Look for: "Investor's Name:", "اسم المستثمر"
    - Extract full name exactly as written
    
    guardian_attorney_name = Guardian/Attorney's Name (if any):
    - Look for: "Guardian/Attorney's Name:", "اسم الوصي/الوكيل/الممثل القانوني"
    - English label: "Guardian/Attorney's Name (if any):"
    - Arabic label: "اسم الوصي/الوكيل/الممثل القانوني"
    - Location: Usually near the top of the form
    - If the field exists but is empty/blank, return: ""
    - If the field has a value, extract the exact name
   
    trading_account_Type = Account Type
    - Look for: "Trading Account Type:", checkbox options
    - Options: نقدي (Cash), هامش (Margin), مشتقات (Derivative)
    - Extract SELECTED type with both languages
    
    trading_account_number = Account Number
    - Look for: "Account Number:", "رقم الحساب"
    - Extract exact number/code
    
    security_name = Security/Stock Name
    - Look for: "Security Name:", "اسم الورقة المالية"
    - Extract exact security name (e.g., "Borouge", "ADNOC", etc.)
    
    security_volume_quantity = Number of Shares/Securities
    - Look for: "Security Volume:", "Quantity:", "عدد الأوراق المالية"
    - Extract numeric value only (e.g., "50000")
    
    security_volume_quantity_unit = Quantity Unit Label
    - Look for: Unit description near quantity field
    - Default: "عدد الأوراق (Quantity)" if not explicitly stated
    
    order_type = Order Type
    - Look for: "Order Type:", "نوع الأمر"
    - Options: 
      * سعر السوق (Market Price)
      * سعر محدد (Limit Price)
      * السعر المرجح (TWAP)
      * نسبة الحجم (POV)
      * نهاية يوم التداول (End Point)
    - Extract SELECTED type with both languages
    
    order_type_price = Order Price
    - Look for: "Price:", "السعر" field value
    - Extract numeric value (e.g., "2.6")
    - Return empty string "" if not applicable (e.g., Market Price orders)
    
    order_validity = Order Validity Period
    - Look for: "Order Validity:", "صلاحية الأمر"
    - Options:
      * يومي (Daily)
      * ساري حتى الإلغاء (GTW)
      * حتى التاريخ (GTD)
      * حتى الإقفال (GTC)
    - Extract SELECTED option with both languages
    
    authorized_signatory_name = Authorized Person Name
    - Look for: Signature section, "Authorized Signatory", employee name
    - Extract full name from signature area
    
    authorized_signatory_code = Employee/Authorization Code
    - Look for: Code/ID near authorized signatory name
    - Extract alphanumeric code (e.g., "ARC-218")
    
    EXTRACTION RULES:
    - If field not found or unclear, return empty string ""
    - Preserve exact formatting and spacing from document
    - For bilingual fields, keep format: "Arabic (English)"
    - For checkboxes, extract only the SELECTED option
    - Numbers should be extracted without thousand separators
    - Times can be 12-hour or 24-hour format as shown
    - Arabic text should be preserved exactly as written
    """
    
    document_images: List[dspy.Image] = dspy.InputField(
        desc="All pages of trading order document as images"
    )
    document_text: str = dspy.InputField(
        desc="Extracted text from all pages of the document"
    )
    special_instructions: str = dspy.InputField(
        desc="Format-specific extraction instructions"
    )
    trading_info: TradingOrderInfo = dspy.OutputField(
        desc="Extracted trading order information in flattened structure"
    )

# ============================================================================
# PDF Processor
# ============================================================================

class PDFProcessor:
    """Handles PDF operations: download, text extraction, image conversion"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to PIL images"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        images = []
        zoom = 200 / 72  # 200 DPI
        
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        doc.close()
        return images
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Convert PDF to images and extract text using Gemini 2.0"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        genai.configure(api_key=Config.GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        
        # Convert PDF to images
        images = PDFProcessor.pdf_to_images(pdf_path)
        logger.info(f"📄 Converted PDF to {len(images)} images")
        
        all_text = []
        
        # Process each page image with Gemini
        for page_num, img in enumerate(images, start=1):
            logger.info(f"🤖 Processing page {page_num}/{len(images)} with Gemini 2.0...")
            
            # prompt = f"""Extract ALL text from this trading order form image exactly as it appears.
            
            # This is a bilingual document (Arabic/English). Extract both languages.
            
            # Output format:
            # === PAGE {page_num} ===
            # [exact text content]

            # Preserve all text, formatting, numbers, checkmarks, and structure. 
            # Include Arabic and English text.
            # Dont missed any content in extraction eg(Guardian/Attorney's Name)
            # Do not summarize."""


            prompt = f"""Extract ALL text from this trading order form image exactly as it appears, page {page_num}.

                This is a bilingual document (Arabic/English). Extract BOTH languages for every field.

                CRITICAL: Do not skip ANY fields, including:
                - Guardian/Attorney's Name (if any)
                - All checkboxes (marked or unmarked)
                - All form fields (filled or empty)
                - All headers and labels
                - All numbers, signatures, and stamps

                Output in clean markdown format following this structure:

                === PAGE {page_num} ===

                # [Company Name/Logo]

                **Market:** السوق (The Market)
                   eg(ADX,DFM)
                   wrong Extraction 'AX' then convert to 'ADX'
                   wrong Extraction 'AFM' then convert to 'DFM' 

                
                **Market:** [value] | **Date:** [value] | **Time:** [value]
                - Numbers: 0(oval), 1(line), 2(curve+line), 3(2curves), 4(triangle), 5(flat+curve), 6(loop-bottom), 7(line±bar), 8(2loops), 9(loop-top)
                
                NUMBER RECOGNITION:
                0 = round oval | 1 = straight line | 2 = curve+line | 3 = two curves
                4 = triangle top | 5 = flat top+curve | 6 = loop bottom | 7 = angled line
                8 = two loops | 9 = loop top+line
                
                
                ## Action Type / نوع العملية
                - [ ] Buy / شراء
                - [✓] Sell / بيع  
                - [ ] Modify / تعديل
                - [ ] Cancel / الغاء

                ## Investor Information

                **Investor's Name / اسم المستثمر:**  
                [exact name as written]

                **Guardian/Attorney's Name (if any) / اسم الوصي/الوكيل/ولي(ان وجد):**  
                [exact name as written]

                ## Trading Account / رقم نوع حساب التداول

                **Type:**
                - [✓] Cash / نقدي
                - [ ] Margin / هامش
                - [ ] Derivative / مشتقات

                **Account Number / رقم الحساب:** [number]

                ## Security Details

                **Security Name / اسم الورقة المالية:** [name]

                **Security Volume / عدد الاوراق المالية:**
                - Quantity / عدد الكمية: [number]
                - Equivalent to / يعادل اسهم الحية: [if filled]

                ## Order Type / نوع الامر

                **Root Price / السعر بسعر:**
                - [ ] Market Price / سعر السوق
                - [✓] Limit Price / سعر محدد

                **Price / السعر:** [value]

                **Floor Price / الارنى السعر:**
                - [ ] TWAP / السعر المعدل(المتوسط)
                - [ ] POV / نسبة الحجم
                - [ ] End Price / نهاية بسعر

                ## Order Validity / صلاحية الامر
                - [✓] Daily / يومي
                - [ ] GTD / محدد بتاريخ
                - [ ] GTDC / حتى الالغاء
                - [ ] GTC / حتى الاقفال

                ## Authorization / التوقيع

                **Signature / Company Stamp / توقيع و ختم الشركة:**  
                [signature details]

                **Name:** [name]  
                **Reference:** [reference number]

                **Note:** [any footer notes]

                ---

                Extract with 100% accuracy. Include every visible character."""
            
            response = model.generate_content([prompt, img])
            page_text = response.text
            all_text.append(page_text)
        
        # Concatenate all pages
        full_text = "\n\n".join(all_text)
        logger.info(f"📝 Extracted {len(full_text)} characters")
        print(f"📝 Extracted:{full_text}")
        
        return full_text
    
    @staticmethod
    def pil_to_dspy_image(pil_img: Image.Image) -> dspy.Image:
        """Convert PIL image to DSPy image format"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            pil_img.save(tmp.name)
            dspy_img = dspy.Image(tmp.name)
        os.unlink(tmp.name)
        return dspy_img

# ============================================================================
# DSPy Module
# ============================================================================

class SimpleTradingExtractor(dspy.Module):
    """DSPy module for trading order extraction"""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TradingOrderExtractor)
    
    def forward(self, document_images: List[dspy.Image], document_text: str, special_instructions: str):
        """Run extraction"""
        return self.generate(
            document_images=document_images,
            document_text=document_text,
            special_instructions=special_instructions
        )

# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages model training, loading, and saving"""
    
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        
        # Initialize DSPy with Gemini 2.5 for mapping/extraction
        # Added error handling for Streamlit compatibility
        try:
            # Check if DSPy is already configured
            if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
                lm = dspy.LM(
                    "gemini/gemini-2.5-flash",
                    api_key=Config.GEMINI_API_KEY,
                    temperature=1,
                    max_tokens=50000,
                    cache=False
                )
                dspy.settings.configure(lm=lm, track_usage=True)
            else:
                # Already configured - reuse existing settings
                logger.info("✅ DSPy already configured, reusing existing settings")
        except RuntimeError as e:
            # Handle DSPy reconfiguration error (common in Streamlit)
            if "dspy.settings" in str(e):
                logger.info("✅ DSPy settings already configured by another instance")
                pass  # Continue - this is expected and safe
            else:
                # Different RuntimeError - re-raise it
                raise
    
    def get_model(
        self,
        training_examples: List[Tuple[str, TradingOrderInfo]],
        force_retrain: bool = False
    ) -> SimpleTradingExtractor:
        """Get model - load existing or train new one"""
        
        os.makedirs(self.models_dir, exist_ok=True)
        model_file = os.path.join(self.models_dir, "trading_extractor.json")
        
        logger.info(f"📁 Models dir: {self.models_dir}")
        logger.info(f"📄 Model file path: {model_file}")
        
        # Check if model exists locally and load it
        if os.path.exists(model_file) and not force_retrain and not os.path.isdir(model_file):
            try:
                return self._load_model(model_file)
            except Exception as e:
                logger.warning(f"🔄 Retraining due to load error: {e}")
                return self._train_model(training_examples)
        
        # Train new model
        return self._train_model(training_examples)
    
    def _load_model(self, model_file: str) -> SimpleTradingExtractor:
        """Load existing model from file"""
        logger.info(f"📂 Loading: {model_file}")
        extractor = SimpleTradingExtractor()
        extractor.load(model_file)
        logger.info("✅ Loaded successfully")
        return extractor
    
    def _train_model(
        self,
        training_examples: List[Tuple[str, TradingOrderInfo]]
    ) -> SimpleTradingExtractor:
        """Train new model with examples"""
        from dspy.teleprompt import LabeledFewShot
        
        logger.info("\n" + "="*70)
        logger.info("🔥 TRAINING TRADING ORDER EXTRACTOR")
        logger.info("="*70)
        
        # Get special instructions
        special_instructions = """Extract information from Al Ramz trading order forms.and mapped value properly in json for new text 
                               1. guardian_attorney_name (MANDATORY FIELD):
                                - English label: "Guardian/Attorney's Name (if any):"
                                - Arabic label: "اسم الوصي/الوكيل/الممثل القانوني"
                                - Location: Usually near the top of the form
                                - If the field exists but is empty/blank, return: ""
                                - If the field has a value, extract the exact name
                                - DO NOT skip this field under any circumstances """
        logger.info(f"\n📝 Special Instructions:\n{special_instructions}\n")
        
        trainset = []
        pdf_processor = PDFProcessor()
        
        for i, (pdf_path, expected_output) in enumerate(training_examples, 1):
            logger.info(f"\n📖 Training example {i}/{len(training_examples)}: {pdf_path}")
            
            if not os.path.exists(pdf_path):
                logger.warning(f"⚠️ Skipping - file not found: {pdf_path}")
                continue
            
            # Extract both images and text using Gemini 2.0
            pil_images = pdf_processor.pdf_to_images(pdf_path)
            dspy_images = [pdf_processor.pil_to_dspy_image(img) for img in pil_images]
            pdf_text = pdf_processor.extract_text_from_pdf(pdf_path)
            
            example = dspy.Example(
                document_images=dspy_images,
                document_text=pdf_text,
                special_instructions=special_instructions,
                trading_info=expected_output
            ).with_inputs("document_images", "document_text", "special_instructions")
            
            logger.info(f"✅ Loaded - {len(pil_images)} pages, {len(pdf_text)} chars")
            trainset.append(example)
        
        if not trainset:
            logger.error("❌ No training data!")
            sys.exit(1)
        
        logger.info(f"\n✅ Training examples loaded: {len(trainset)}")
        
        # Train model with Gemini 2.5
        optimizer = LabeledFewShot(k=len(trainset))
        extractor = SimpleTradingExtractor()
        
        logger.info(f"🔧 Compiling with LabeledFewShot (k={len(trainset)})...")
        compiled_extractor = optimizer.compile(extractor, trainset=trainset)
        
        # Save model locally
        os.makedirs(self.models_dir, exist_ok=True)
        model_file = os.path.join(self.models_dir, "trading_extractor.json")
        
        # Remove directory if it exists at model_file path
        if os.path.exists(model_file) and os.path.isdir(model_file):
            logger.warning(f"⚠️ {model_file} is a directory, removing it")
            import shutil
            shutil.rmtree(model_file)
        
        compiled_extractor.save(model_file)
        
        logger.info(f"\n💾 Saved locally to: {model_file}")
        logger.info("✅ Training complete!")
        logger.info("="*70)
        
        return compiled_extractor

# ============================================================================
# Trading Order Extractor Service
# ============================================================================

class TradingOrderExtractorService:
    """Main service for trading order extraction"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.pdf_processor = PDFProcessor()
    
    def extract_trading_order(
        self,
        pdf_path: str,
        extractor: SimpleTradingExtractor,
        clean_text: str = None
    ) -> Optional[Dict[str, Any]]:
        """Extract trading order information from PDF using trained model"""
        logger.info(f"\n🔍 Extracting trading order from: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error("❌ File not found")
            return None
        
        # Get special instructions
        special_instructions = "Extract from Al Ramz trading order form."
        
        # Extract both images and text
        pil_images = self.pdf_processor.pdf_to_images(pdf_path)
        dspy_images = [self.pdf_processor.pil_to_dspy_image(img) for img in pil_images]
        
        # Use provided text or extract new
        if clean_text is None:
            clean_text = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        logger.info(f"📄 Text preview (first 300 chars):\n{clean_text[:500]}...")
        
        # Run extraction with Gemini 2.5
        result = extractor(
            document_images=dspy_images,
            document_text=clean_text,
            special_instructions=special_instructions
        )
        
        # Get token usage
        usage = result.get_lm_usage()
        model_name = list(usage.keys())[0]
        tokens = usage[model_name]

        dspy_input_tokens = tokens['prompt_tokens']
        dspy_output_tokens = tokens['completion_tokens']
        dspy_total_tokens = tokens['total_tokens']
        return {
            "broker": result.trading_info.broker,
            "market": result.trading_info.market,
            "date": result.trading_info.date,
            "time": result.trading_info.time,
            "action_type": result.trading_info.action_type,
            "investor_name": result.trading_info.investor_name,
            "guardian_attorney_name": result.trading_info.guardian_attorney_name,
            "trading_account_Type": result.trading_info.trading_account_Type,
            "trading_account_number": result.trading_info.trading_account_number,
            "security_name": result.trading_info.security_name,
            "security_volume_quantity": result.trading_info.security_volume_quantity,
            "security_volume_quantity_unit": result.trading_info.security_volume_quantity_unit,
            "order_type": result.trading_info.order_type,
            "order_type_price": result.trading_info.order_type_price,
            "order_validity": result.trading_info.order_validity,
            "authorized_signatory_name": result.trading_info.authorized_signatory_name,
            "authorized_signatory_code": result.trading_info.authorized_signatory_code,
            "combined_input_token": dspy_input_tokens,
            "combined_output_token": dspy_output_tokens,
            "combined_total_token": dspy_total_tokens
        }
    
    @staticmethod
    def convert_dict_to_training_examples(
        trading_examples_dict: Dict[str, Any]
    ) -> List[Tuple[str, TradingOrderInfo]]:
        """Convert trading order examples dictionary to training format"""
        training_examples = []
        
        for example_key, example_data in trading_examples_dict.items():
            if not example_data:
                continue
            
            trading_info = TradingOrderInfo(
                broker=example_data.get('broker', ''),
                market=example_data.get('market', ''),
                date=example_data.get('date', ''),
                time=example_data.get('time', ''),
                action_type=example_data.get('action_type', ''),
                investor_name=example_data.get('investor_name', ''),
                guardian_attorney_name=example_data.get('guardian_attorney_name', ''),
                trading_account_Type=example_data.get('trading_account_Type', ''),
                trading_account_number=example_data.get('trading_account_number', ''),
                security_name=example_data.get('security_name', ''),
                security_volume_quantity=example_data.get('security_volume_quantity', ''),
                security_volume_quantity_unit=example_data.get('security_volume_quantity_unit', ''),
                order_type=example_data.get('order_type', ''),
                order_type_price=example_data.get('order_type_price', ''),
                order_validity=example_data.get('order_validity', ''),
                authorized_signatory_name=example_data.get('authorized_signatory_name', ''),
                authorized_signatory_code=example_data.get('authorized_signatory_code', '')
            )
            
            pdf_path = example_data.get('pdf_path', f'pdf/{example_key}.pdf')
            training_examples.append((pdf_path, trading_info))
        
        return training_examples

# ============================================================================
# Main Execution
# ============================================================================
def remove_whitespace_from_json(json_data):
    """
    Remove leading/trailing whitespace and newlines from JSON string values
    """
    if isinstance(json_data, dict):
        return {key: remove_whitespace_from_json(value) for key, value in json_data.items()}
    elif isinstance(json_data, list):
        return [remove_whitespace_from_json(item) for item in json_data]
    elif isinstance(json_data, str):
        # Remove leading/trailing whitespace and newlines
        return json_data.strip()
    else:
        return json_data
    
def main_trading_extraction(
    pdf_path: str,
    trading_examples: Dict[str, Any],
    force_retrain: bool = False,
    extracted_text: str = None
):
    """
    Main function to extract trading order information
    
    Args:
        pdf_path: Path to the PDF file to extract
        trading_examples: Dictionary of training examples
        force_retrain: Whether to force model retraining
        extracted_text: Pre-extracted text (optional)
    
    Returns:
        Dictionary with extracted information and token usage
    """
    
    # Initialize service
    service = TradingOrderExtractorService()
    
    # Convert examples to training format
    training_examples = service.convert_dict_to_training_examples(trading_examples)
    
    logger.info(f"\n🎯 Processing {len(training_examples)} training examples")
    
    # Get or train model
    extractor = service.model_manager.get_model(
        training_examples,
        force_retrain=force_retrain
    )
    
    # Extract from new PDF
    result = service.extract_trading_order(
        pdf_path,
        extractor,
        extracted_text
    )
    result1 = remove_whitespace_from_json(result)
    if result1:
        combined_input = result1.get("combined_input_token")
        combined_output = result1.get("combined_output_token")
        combined_total = result1.get("combined_total_token")
        
        print("\n" + "="*70)
        print("📊 TRADING ORDER EXTRACTION RESULT")
        print("="*70)
        print(json.dumps(result1, indent=2, ensure_ascii=False))
        print("="*70)
        print(f"\n💰 Token Usage:")
        print(f"   Input:  {combined_input}")
        print(f"   Output: {combined_output}")
        print(f"   Total:  {combined_total}")
        print("="*70)
        
        return result1, combined_input, combined_output, combined_total
    
    return None, 0, 0, 0

# ============================================================================
# Example Usage
# ============================================================================

# if __name__ == "__main__":
#     # Example training data
#     example_data = {
#         'example_1': {
#             "pdf_path": "FirstOrderSlip.pdf",
#             "broker": "Al Ramz",
#             "market": "السوق (ADX)",
#             "date": "4/7/21",
#             "time": "9:32",
#             "action_type": "شراء (Buy)",
#             "investor_name": "Mim Okta Mininnic Co.",
#             "guardian_attorney_name": "محمد عبد المالك",
#             "trading_account_Type": "نقدي (Cash)",
#             "trading_account_number": "105",
#             "security_name": "Borouge",
#             "security_volume_quantity": "50000",
#             "security_volume_quantity_unit": "عدد الأوراق (Quantity)",
#             "order_type": "سعر محدد (Limit Price)",
#             "order_type_price": "2.6",
#             "order_validity": "يومي (Daily)",
#             "authorized_signatory_name": "Rahal Kamarji",
#             "authorized_signatory_code": "ARC-218"
#         }
#     }
    
#     # Run extraction
#     result, input_tokens, output_tokens, total_tokens = main_trading_extraction(
#         pdf_path="50023807-WO55-Lulu-2.pdf",
#         trading_examples=example_data,
#         force_retrain=False
#     )