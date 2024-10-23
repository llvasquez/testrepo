import io
from typing import List, Tuple, Optional
import logging
from pathlib import Path

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

def main():
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([.5, .5])

    try:
        # Load all models
        models = load_models()
    except SuryaProcessingError as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    # File upload and language selection
    in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
    languages = st.sidebar.multiselect(
        "Languages", 
        sorted(list(surya_modules["CODE_TO_LANGUAGE"].values())), 
        default=[], 
        max_selections=4,
        help="Select document languages (optional) to improve OCR accuracy."
    )

    if in_file is None:
        st.stop()

    try:
        # Process file based on type
        filetype = in_file.type
        if "pdf" in filetype:
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
        else:
            pil_image = Image.open(in_file).convert("RGB")
            pil_image_highres = pil_image
            page_number = None

        # Processing options
        use_pdf_boxes = st.sidebar.checkbox("Use PDF table boxes", value=True,
                                          help="Use PDF file bounding boxes vs. text detection model for tables")
        skip_table_detection = st.sidebar.checkbox("Skip table detection", value=False,
                                                 help="Treat the whole image/page as a table")

        # Process document when requested
        if st.sidebar.button("Process Document"):
            with st.spinner("Processing document..."):
                results = process_document(
                    models=models,
                    image=pil_image,
                    highres_image=pil_image_highres,
                    filepath=in_file if "pdf" in filetype else None,
                    page_idx=page_number - 1 if page_number else None,
                    languages=languages,
                    use_pdf_boxes=use_pdf_boxes,
                    skip_table_detection=skip_table_detection
                )

                # Display results in tabs
                with col1:
                    tabs = st.tabs(["Text Detection", "Layout", "OCR", "Reading Order", "Table"])
                    
                    with tabs[0]:
                        if "text_detection" in results:
                            st.image(results["text_detection"]["image"], caption="Detected Text", use_column_width=True)
                            st.json(results["text_detection"]["prediction"].model_dump(
                                exclude=["heatmap", "affinity_map"]), expanded=False)
                    
                    with tabs[1]:
                        if "layout" in results:
                            st.image(results["layout"]["image"], caption="Detected Layout", use_column_width=True)
                            st.json(results["layout"]["prediction"].model_dump(
                                exclude=["segmentation_map"]), expanded=False)
                    
                    with tabs[2]:
                        if "ocr" in results:
                            st.image(results["ocr"]["image"], caption="OCR Result", use_column_width=True)
                            json_tab, text_tab = st.tabs(["JSON", "Text Lines"])
                            with json_tab:
                                st.json(results["ocr"]["prediction"].model_dump(), expanded=False)
                            with text_tab:
                                st.text("\n".join([p.text for p in results["ocr"]["prediction"].text_lines]))
                    
                    with tabs[3]:
                        if "order" in results:
                            st.image(results["order"]["image"], caption="Reading Order", use_column_width=True)
                            st.json(results["order"]["prediction"].model_dump(), expanded=False)
                    
                    with tabs[4]:
                        if "table" in results:
                            st.image(results["table"]["image"], caption="Table Recognition", use_column_width=True)
                            st.json([p.model_dump() for p in results["table"]["prediction"]], expanded=False)

        # Always show original image in second column
        with col2:
            st.image(pil_image, caption="Original Image", use_column_width=True)

    except SuryaProcessingError as e:
        st.error(f"Error processing document: {str(e)}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        logger.exception("Unexpected error occurred")

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

if __name__ == "__main__":
    main()