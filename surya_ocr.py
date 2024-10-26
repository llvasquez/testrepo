from datetime import datetime
import io
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import zipfile
import tempfile
import os
import re
import pypdfium2
import streamlit as st
from pypdfium2 import PdfiumError
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define custom exceptions
class SuryaImportError(Exception):
    """Raised when there are issues importing Surya modules"""
    pass

class SuryaProcessingError(Exception):
    """Raised when there are issues processing images or PDFs"""
    pass

def import_surya_modules():
    """
    Safely import all required Surya modules with proper error handling
    Returns a dictionary of imported modules or None if import fails
    """
    try:
        from surya.detection import batch_text_detection
        from surya.layout import batch_layout_detection
        from surya.model.detection.model import load_model, load_processor
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        from surya.model.ordering.processor import load_processor as load_order_processor
        from surya.model.ordering.model import load_model as load_order_model
        from surya.model.table_rec.model import load_model as load_table_model
        from surya.model.table_rec.processor import load_processor as load_table_processor
        from surya.ordering import batch_ordering
        from surya.postprocessing.heatmap import draw_polys_on_image, draw_bboxes_on_image
        from surya.ocr import run_ocr
        from surya.postprocessing.text import draw_text_on_image
        from surya.languages import CODE_TO_LANGUAGE
        from surya.input.langs import replace_lang_with_code
        from surya.schema import OCRResult, TextDetectionResult, LayoutResult, OrderResult, TableResult
        from surya.settings import settings
        from surya.tables import batch_table_recognition
        from surya.postprocessing.util import rescale_bboxes, rescale_bbox
        
        # Try to import pdflines separately as it seems to be problematic
        try:
            from surya.input.pdflines import get_page_text_lines, get_table_blocks
            has_pdflines = True
        except ImportError:
            logger.warning("Could not import pdflines module. Table recognition from PDFs may be limited.")
            has_pdflines = False
            
            # Define dummy functions if pdflines is not available
            def get_page_text_lines(*args, **kwargs):
                return [[]]
                
            def get_table_blocks(*args, **kwargs):
                return []

        return {
            "batch_text_detection": batch_text_detection,
            "batch_layout_detection": batch_layout_detection,
            "load_model": load_model,
            "load_processor": load_processor,
            "load_rec_model": load_rec_model,
            "load_rec_processor": load_rec_processor,
            "load_order_processor": load_order_processor,
            "load_order_model": load_order_model,
            "load_table_model": load_table_model,
            "load_table_processor": load_table_processor,
            "batch_ordering": batch_ordering,
            "draw_polys_on_image": draw_polys_on_image,
            "draw_bboxes_on_image": draw_bboxes_on_image,
            "run_ocr": run_ocr,
            "draw_text_on_image": draw_text_on_image,
            "CODE_TO_LANGUAGE": CODE_TO_LANGUAGE,
            "replace_lang_with_code": replace_lang_with_code,
            "settings": settings,
            "batch_table_recognition": batch_table_recognition,
            "rescale_bboxes": rescale_bboxes,
            "rescale_bbox": rescale_bbox,
            "get_page_text_lines": get_page_text_lines,
            "get_table_blocks": get_table_blocks,
            "has_pdflines": has_pdflines
        }
    except ImportError as e:
        logger.error(f"Failed to import Surya modules: {str(e)}")
        raise SuryaImportError(f"Failed to import required Surya modules. Please ensure Surya is installed correctly: {str(e)}")

# Load all required modules
try:
    surya_modules = import_surya_modules()
except SuryaImportError as e:
    st.error(f"Error: {str(e)}")
    st.error("Please make sure you have installed surya-ocr correctly:")
    st.code("pip install surya-ocr --upgrade")
    st.stop()

@st.cache_resource
def load_models():
    """Load all required models with proper error handling"""
    try:
        det_model, det_processor = (
            surya_modules["load_model"](checkpoint=surya_modules["settings"].DETECTOR_MODEL_CHECKPOINT),
            surya_modules["load_processor"](checkpoint=surya_modules["settings"].DETECTOR_MODEL_CHECKPOINT)
        )
        rec_model, rec_processor = surya_modules["load_rec_model"](), surya_modules["load_rec_processor"]()
        layout_model, layout_processor = (
            surya_modules["load_model"](checkpoint=surya_modules["settings"].LAYOUT_MODEL_CHECKPOINT),
            surya_modules["load_processor"](checkpoint=surya_modules["settings"].LAYOUT_MODEL_CHECKPOINT)
        )
        order_model, order_processor = surya_modules["load_order_model"](), surya_modules["load_order_processor"]()
        table_model, table_processor = surya_modules["load_table_model"](), surya_modules["load_table_processor"]()
        
        return {
            "det": (det_model, det_processor),
            "rec": (rec_model, rec_processor),
            "layout": (layout_model, layout_processor),
            "order": (order_model, order_processor),
            "table": (table_model, table_processor)
        }
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise SuryaProcessingError(f"Failed to load required models: {str(e)}")

def extract_structured_text(ocr_result) -> dict:
    """
    Extract structured information from OCR results with improved pattern matching.
    Returns a dictionary with DateofBirth, Name, and Description fields.
    """
    if not ocr_result or not hasattr(ocr_result, 'text_lines'):
        return {
            "DateofBirth": None,
            "Name": None,
            "Description": None
        }

    # Join all text lines with proper spacing
    all_text = ' '.join([line.text.strip() for line in ocr_result.text_lines if line.text])
    
    # Initialize result dictionary
    result = {
        "DateofBirth": None,
        "Name": None,
        "Description": None
    }
    
    # Enhanced date patterns
    date_patterns = [
        r'\b(?:DOB|Date\s+of\s+Birth|Birth\s+Date)[\s:]+([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b',
        r'\b(?:DOB|Date\s+of\s+Birth|Birth\s+Date)[\s:]+([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})\b',
        r'DATE OF BIRTH:\s*([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})',  # Specific to Beanie Baby format
        r'\b([0-9]{1,2}[-/][0-9]{1,2}[-/][0-9]{2,4})\b',
        r'\b([A-Za-z]+\s+[0-9]{1,2},?\s+[0-9]{4})\b'
    ]
    
    # Try to find date of birth
    for pattern in date_patterns:
        match = re.search(pattern, all_text, re.IGNORECASE)
        if match:
            result["DateofBirth"] = match.group(1) if len(match.groups()) > 0 else match.group(0)
            break
    
    # Enhanced name patterns specifically for Beanie Babies
    name_patterns = [
        r'\b([A-Za-z]+)™\b',  # Single word followed by ™ symbol
        r'\b([A-Za-z]+)™(?=\s|$)',  # Word with ™ at end of string or before space
        r'([A-Za-z]+)[™®]'  # Fallback: word followed by ™ or ® without word boundary
    ]
    
    # Try to find name
    for pattern in name_patterns:
        match = re.search(pattern, all_text)
        if match:
            name = match.group(1).strip()
            # Remove any trailing punctuation or symbols
            name = re.sub(r'[^\w\s.-]$', '', name)
            # Skip if the name is part of "The Beanie Babies Collection"
            if name.lower() not in ['the', 'beanie', 'babies', 'collection']:
                result["Name"] = name
                break
    
    # Extract description - specifically looking for the poem/description on Beanie Baby tags
    description_patterns = [
        r"Isn't this just.*?\n(.*?)(?=www\.ty\.com|$)",  # Specific to this format
        r"When we saw.*?\n(.*?)(?=www\.ty\.com|$)",      # Alternative starting point
        r'(.+?)(?=www\.ty\.com|$)'                       # Fallback pattern
    ]
    
    for pattern in description_patterns:
        match = re.search(pattern, all_text, re.DOTALL | re.IGNORECASE)
        if match:
            description = match.group(1).strip()
            # Clean up the description
            description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
            if description:
                result["Description"] = description
                break

    return result

def process_document(models: dict, image: Image.Image, highres_image: Optional[Image.Image] = None,
                    filepath: Optional[str] = None, page_idx: Optional[int] = None,
                    languages: List[str] = None, use_pdf_boxes: bool = True,
                    skip_table_detection: bool = False) -> dict:
    """
    Process a document with all available methods
    Returns a dictionary containing all results and processed images
    """
    results = {}
    
    try:
        # Text Detection
        det_pred = surya_modules["batch_text_detection"]([image], models["det"][0], models["det"][1])[0]
        polygons = [p.polygon for p in det_pred.bboxes]
        det_img = surya_modules["draw_polys_on_image"](polygons, image.copy())
        results["text_detection"] = {"image": det_img, "prediction": det_pred}

        # Layout Detection
        layout_pred = surya_modules["batch_layout_detection"]([image], models["layout"][0], models["layout"][1], [det_pred])[0]
        polygons = [p.polygon for p in layout_pred.bboxes]
        labels = [p.label for p in layout_pred.bboxes]
        layout_img = surya_modules["draw_polys_on_image"](polygons, image.copy(), labels=labels, label_font_size=18)
        results["layout"] = {"image": layout_img, "prediction": layout_pred}

        # OCR
        if languages:
            surya_modules["replace_lang_with_code"](languages)
            ocr_pred = surya_modules["run_ocr"]([image], [languages], models["det"][0], models["det"][1],
                                              models["rec"][0], models["rec"][1], highres_images=[highres_image or image])[0]
            bboxes = [l.bbox for l in ocr_pred.text_lines]
            text = [l.text for l in ocr_pred.text_lines]
            rec_img = surya_modules["draw_text_on_image"](bboxes, text, image.size, languages, has_math="_math" in languages)
            results["ocr"] = {"image": rec_img, "prediction": ocr_pred}

        # Reading Order
        bboxes = [l.bbox for l in layout_pred.bboxes]
        order_pred = surya_modules["batch_ordering"]([image], [bboxes], models["order"][0], models["order"][1])[0]
        polys = [l.polygon for l in order_pred.bboxes]
        positions = [str(l.position) for l in order_pred.bboxes]
        order_img = surya_modules["draw_polys_on_image"](polys, image.copy(), labels=positions, label_font_size=18)
        results["order"] = {"image": order_img, "prediction": order_pred}

        # Table Recognition
        if skip_table_detection:
            layout_tables = [(0, 0, highres_image.size[0], highres_image.size[1])]
            table_imgs = [highres_image or image]
        else:
            layout_tables_lowres = [l.bbox for l in layout_pred.bboxes if l.label == "Table"]
            table_imgs = []
            layout_tables = []
            for tb in layout_tables_lowres:
                highres_bbox = surya_modules["rescale_bbox"](tb, image.size, (highres_image or image).size)
                table_imgs.append((highres_image or image).crop(highres_bbox))
                layout_tables.append(highres_bbox)

        if filepath and surya_modules["has_pdflines"]:
            try:
                page_text = surya_modules["get_page_text_lines"](filepath, [page_idx], [(highres_image or image).size])[0]
                table_bboxes = surya_modules["get_table_blocks"](layout_tables, page_text, (highres_image or image).size)
            except PdfiumError:
                table_bboxes = [[] for _ in layout_tables]
        else:
            table_bboxes = [[] for _ in layout_tables]

        if not use_pdf_boxes or any(len(tb) == 0 for tb in table_bboxes):
            det_results = surya_modules["batch_text_detection"](table_imgs, models["det"][0], models["det"][1])
            table_bboxes = [[{"bbox": tb.bbox, "text": None} for tb in det_result.bboxes] for det_result in det_results]

        table_preds = surya_modules["batch_table_recognition"](table_imgs, table_bboxes, models["table"][0], models["table"][1])
        table_img = image.copy()

        for results_table, table_bbox in zip(table_preds, layout_tables):
            adjusted_bboxes = []
            labels = []
            colors = []

            for item in results_table.rows + results_table.cols:
                adjusted_bboxes.append([
                    (item.bbox[0] + table_bbox[0]),
                    (item.bbox[1] + table_bbox[1]),
                    (item.bbox[2] + table_bbox[0]),
                    (item.bbox[3] + table_bbox[1])
                ])
                labels.append(item.label)
                colors.append("blue" if hasattr(item, "row_id") else "red")
                
            table_img = surya_modules["draw_bboxes_on_image"](
                adjusted_bboxes, highres_image or image, 
                labels=labels, label_font_size=18, color=colors
            )
        results["table"] = {"image": table_img, "prediction": table_preds}

        return results

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise SuryaProcessingError(f"Error processing document: {str(e)}")

def is_valid_image(file_path: str) -> bool:
    """Check if the file is a valid image format"""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    return Path(file_path).suffix.lower() in valid_extensions

def process_uploaded_zip(zip_file) -> List[Tuple[str, Image.Image]]:
    """Process uploaded ZIP file containing images"""
    images = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(temp_dir)
            
            # Walk through all files in the extracted directory
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if is_valid_image(file):
                        file_path = os.path.join(root, file)
                        try:
                            img = Image.open(file_path).convert("RGB")
                            images.append((file, img))
                        except Exception as e:
                            logger.warning(f"Failed to process image {file}: {str(e)}")
    
    return sorted(images, key=lambda x: x[0].lower())

def process_uploaded_files(files) -> List[Tuple[str, Image.Image]]:
    """Process uploaded files from a folder"""
    images = []
    for file in files:
        if is_valid_image(file.name):
            try:
                img = Image.open(file).convert("RGB")
                images.append((file.name, img))
            except Exception as e:
                logger.warning(f"Failed to process image {file.name}: {str(e)}")
    
    return sorted(images, key=lambda x: x[0].lower())

def open_pdf(pdf_file):
    """Open a PDF file using pypdfium2"""
    try:
        stream = io.BytesIO(pdf_file.getvalue())
        return pypdfium2.PdfDocument(stream)
    except Exception as e:
        raise SuryaProcessingError(f"Error opening PDF file: {str(e)}")

def get_page_image(pdf_file, page_num: int, dpi: int) -> Image.Image:
    """Get a specific page from a PDF file as an image"""
    try:
        doc = open_pdf(pdf_file)
        renderer = doc.render(
            pypdfium2.PdfBitmap.to_pil,
            page_indices=[page_num - 1],
            scale=dpi / 72,
        )
        png = list(renderer)[0]
        return png.convert("RGB")
    except Exception as e:
        raise SuryaProcessingError(f"Error extracting page image: {str(e)}")
        
def main():
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([.5, .5])

    try:
        # Load all models
        models = load_models()
    except SuryaProcessingError as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    # Upload type selection
    upload_type = st.sidebar.radio(
        "Upload type:",
        ["Single File", "Image Folder", "ZIP File"],
        help="Choose between uploading a single file, folder, or ZIP file containing multiple images"
    )

    # Initialize variables
    current_images = []
    current_highres_images = []
    page_number = None
    in_file = None

    # Handle file uploads based on type
    if upload_type == "Single File":
        in_file = st.sidebar.file_uploader(
            "PDF file or image:", 
            type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
        )
        if in_file is None:
            st.info("Please upload a file")
            st.stop()
            
        try:
            if "pdf" in in_file.type:
                doc = open_pdf(in_file)
                page_count = len(doc)
                page_number = st.sidebar.number_input(
                    f"Page number (1-{page_count}):",
                    min_value=1,
                    value=1,
                    max_value=page_count
                )
                pil_image = get_page_image(in_file, page_number, surya_modules["settings"].IMAGE_DPI)
                pil_image_highres = get_page_image(in_file, page_number, surya_modules["settings"].IMAGE_DPI_HIGHRES)
                current_images = [(f"page_{page_number}", pil_image)]
                current_highres_images = [(f"page_{page_number}", pil_image_highres)]
            else:
                pil_image = Image.open(in_file).convert("RGB")
                current_images = [(in_file.name, pil_image)]
                current_highres_images = [(in_file.name, pil_image)]
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()

    elif upload_type == "ZIP File":
        in_file = st.sidebar.file_uploader(
            "ZIP file containing images:",
            type=["zip"]
        )
        if in_file is None:
            st.info("Please upload a ZIP file")
            st.stop()
            
        try:
            current_images = process_uploaded_zip(in_file)
            current_highres_images = current_images
            if not current_images:
                st.error("No valid images found in the ZIP file")
                st.stop()
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")
            st.stop()

    else:  # Image Folder
        in_files = st.sidebar.file_uploader(
            "Upload image files:",
            type=["png", "jpg", "jpeg", "gif", "webp"],
            accept_multiple_files=True
        )
        if not in_files:
            st.info("Please upload image files")
            st.stop()
            
        try:
            current_images = process_uploaded_files(in_files)
            current_highres_images = current_images
            if not current_images:
                st.error("No valid images uploaded")
                st.stop()
        except Exception as e:
            st.error(f"Error processing uploaded files: {str(e)}")
            st.stop()

    # Language selection
    languages = st.sidebar.multiselect(
        "Languages", 
        sorted(list(surya_modules["CODE_TO_LANGUAGE"].values())), 
        default=[], 
        max_selections=4,
        help="Select document languages (optional) to improve OCR accuracy."
    )

    # Image selection for multiple files
    if upload_type in ["ZIP File", "Image Folder"] and current_images:
        # Add "Select All" option at the top of the selection list
        image_options = ["All Images"] + [img[0] for img in current_images]
        selected_image = st.sidebar.selectbox(
            "Select image to process:",
            options=image_options,
            format_func=lambda x: x
        )
        
        # Handle image selection
        if selected_image == "All Images":
            # Keep all images for processing
            selected_images = current_images
            selected_highres_images = current_highres_images
        else:
            # Get the selected single image
            current_idx = [img[0] for img in current_images].index(selected_image)
            selected_images = [current_images[current_idx]]
            selected_highres_images = [current_highres_images[current_idx]]
    else:
        selected_images = current_images
        selected_highres_images = current_highres_images

    # Processing options
    use_pdf_boxes = st.sidebar.checkbox(
        "Use PDF table boxes",
        value=True,
        help="Use PDF file bounding boxes vs. text detection model for tables"
    )
    skip_table_detection = st.sidebar.checkbox(
        "Skip table detection",
        value=False,
        help="Treat the whole image/page as a table"
    )

    # Display total number of images found
    if upload_type in ["ZIP File", "Image Folder"]:
        st.sidebar.info(f"Total images found: {len(current_images)}")
        if selected_image == "All Images":
            st.sidebar.warning(f"Processing all {len(current_images)} images")

    # Display preview of first/selected image in second column
    if selected_images:
        with col2:
            st.image(selected_images[0][1], caption="Preview Image", use_column_width=True)

    # Process document when requested
    if st.sidebar.button("Process Document"):
        # Create a progress bar for multiple images
        progress_bar = st.progress(0)
        
        with st.spinner("Processing document(s)..."):
            try:
                # Create a container for all results
                all_results = {}
                
                # Process each image in the selection
                for idx, ((img_name, pil_image), (_, pil_image_highres)) in enumerate(zip(selected_images, selected_highres_images)):
                    st.subheader(f"Processing: {img_name}")
                    
                    results = process_document(
                        models=models,
                        image=pil_image,
                        highres_image=pil_image_highres,
                        filepath=in_file if upload_type == "Single File" and "pdf" in in_file.type else None,
                        page_idx=page_number - 1 if page_number else None,
                        languages=languages,
                        use_pdf_boxes=use_pdf_boxes,
                        skip_table_detection=skip_table_detection
                    )
                    
                    # Extract just the text information
                    extracted_text = extract_structured_text(results.get("ocr", {}).get("prediction"))
                    all_results[img_name] = extracted_text
                    
                    # Update progress bar
                    progress = (idx + 1) / len(selected_images)
                    progress_bar.progress(progress)

                    # Display results in tabs for current image
                    with col1:
                        tabs = st.tabs(["Extracted Text"])
                        
                        with tabs[0]:
                            if extracted_text:
                                st.json({
                                    "DateofBirth": extracted_text["DateofBirth"],
                                    "Name": extracted_text["Name"],
                                    "Description": extracted_text["Description"]
                                })

                # After processing all images, show combined results
                if len(selected_images) > 1:
                    st.subheader("Combined Results")
                    combined_json = {
                        "processed_at": datetime.now().isoformat(),
                        "total_images": len(selected_images),
                        "images": {
                            name: {
                                "DateofBirth": results["DateofBirth"],
                                "Name": results["Name"],
                                "Description": results["Description"]
                            }
                            for name, results in all_results.items()
                        }
                    }
                    st.json(combined_json)
                else:
                    # For single image
                    st.subheader("Results")
                    final_json = {
                        "processed_at": datetime.now().isoformat(),
                        "results": {
                            "DateofBirth": all_results[list(all_results.keys())[0]]["DateofBirth"],
                            "Name": all_results[list(all_results.keys())[0]]["Name"],
                            "Description": all_results[list(all_results.keys())[0]]["Description"]
                        }
                    }
                    st.json(final_json)

            except SuryaProcessingError as e:
                st.error(f"Error processing document: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                logger.exception("Unexpected error occurred")
            finally:
                # Ensure progress bar reaches 100%
                progress_bar.progress(1.0)
                
if __name__ == "__main__":
    main()
