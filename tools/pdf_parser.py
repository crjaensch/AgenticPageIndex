from pathlib import Path
from typing import Dict, Any
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import get_page_tokens, get_pdf_name

def pdf_parser_tool(context: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
    """
    Extract text, metadata, and token counts from PDF
    
    Args:
        context: Serialized PageIndexContext
        pdf_path: Path to PDF file
        
    Returns:
        Updated context with pdf_metadata and pages_file populated
    """
    
    try:
        # Reconstruct context using consistent approach
        context_obj = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context_obj.config.global_config.log_dir) / context_obj.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context_obj.log_step("pdf_parser", "started", {"pdf_path": pdf_path})
        
        # Validate PDF file
        if not Path(pdf_path).exists():
            raise PageIndexToolError(f"PDF file not found: {pdf_path}")
        
        # Extract PDF metadata
        pdf_name = get_pdf_name(pdf_path)
        context_obj.pdf_metadata = {
            "pdf_name": pdf_name,
            "pdf_path": pdf_path
        }
        
        # Get configuration values
        pdf_parser_type = context_obj.config.pdf_parser.pdf_parser
        model = context_obj.config.global_config.model
        
        # Extract pages with token counts
        pages = get_page_tokens(
            pdf_path, 
            model=model,
            pdf_parser=pdf_parser_type
        )
        
        # Save pages to file
        context_obj.save_pages(pages, log_dir)
        
        # Update metadata
        context_obj.pdf_metadata.update({
            "total_pages": len(pages),
            "total_tokens": sum(page[1] for page in pages)
        })
        
        # Log success
        context_obj.log_step("pdf_parser", "completed", {
            "pages_extracted": len(pages),
            "total_tokens": context_obj.pdf_metadata["total_tokens"]
        })
        
        # Save checkpoint
        context_obj.save_checkpoint(log_dir)
        
        return {
            "success": True,
            "context": context_obj.to_dict(),
            "confidence": 1.0,
            "metrics": {
                "pages_extracted": len(pages),
                "total_tokens": context_obj.pdf_metadata["total_tokens"]
            },
            "errors": [],
            "suggestions": []
        }
        
    except Exception as e:
        # Save failure state
        if 'context_obj' in locals():
            context_obj.log_step("pdf_parser", "failed", {"error": str(e)})
            context_obj.save_checkpoint(log_dir)
            
        raise PageIndexToolError(f"PDF parsing failed: {str(e)}") from e