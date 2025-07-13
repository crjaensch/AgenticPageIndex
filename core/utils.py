import tiktoken
import logging
import os
import json
import PyPDF2
import pymupdf
from io import BytesIO
from typing import List, Tuple, Dict, Any, Union

def count_tokens(text: str, model: str) -> int:
    """Count tokens in text using tiktoken"""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def get_page_tokens(pdf_path: Union[str, BytesIO], model: str = "gpt-4o-mini", 
                   pdf_parser: str = "PyMuPDF") -> List[Tuple[str, int]]:
    """Extract pages with token counts from PDF"""
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = count_tokens(page_text, model)
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            doc = pymupdf.open(stream=pdf_path, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        else:
            raise ValueError(f"Invalid PDF path: {pdf_path}")
        
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = count_tokens(page_text, model)
            page_list.append((page_text, token_length))
        doc.close()
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

def get_pdf_name(pdf_path: Union[str, BytesIO]) -> str:
    """Extract PDF name from path or metadata"""
    if isinstance(pdf_path, str):
        return os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            meta = pdf_reader.metadata
            pdf_name = meta.title if meta and meta.title else 'Untitled'
            return sanitize_filename(pdf_name)
        except:
            return 'Untitled'
    else:
        return 'Unknown'

def sanitize_filename(filename: str, replacement: str = '-') -> str:
    """Sanitize filename for filesystem compatibility"""
    return filename.replace('/', replacement).replace('\\', replacement)

def extract_json(content: str) -> Dict[str, Any]:
    """Extract JSON from LLM response with error handling"""
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}

def get_json_content(response: str) -> str:
    """Extract JSON content from markdown-formatted response"""
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    return response.strip()

def create_recovery_suggestions(error_type: str, context: str) -> List[str]:
    """Generate recovery suggestions based on error type and context"""
    suggestions = []
    
    if "pdf" in error_type.lower():
        suggestions.extend([
            "Verify PDF file exists and is readable",
            "Try using a different PDF parser (PyPDF2 vs PyMuPDF)",
            "Check if PDF is password protected or corrupted"
        ])
    elif "toc" in error_type.lower():
        suggestions.extend([
            "Try processing without TOC detection",
            "Manually specify TOC pages if known",
            "Increase toc_check_page_num if TOC appears later in document"
        ])
    elif "structure" in error_type.lower():
        suggestions.extend([
            "Reduce max_token_num_each_node for smaller chunks",
            "Try different extraction strategy",
            "Check if document has clear section headers"
        ])
    elif "verification" in error_type.lower():
        suggestions.extend([
            "Lower accuracy_threshold in configuration",
            "Skip verification step if structure looks reasonable",
            "Manual review of extracted structure recommended"
        ])
    
    return suggestions
