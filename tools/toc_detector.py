import openai
from pathlib import Path
from typing import Dict, Any, List
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import extract_json, create_recovery_suggestions

def toc_detector_tool(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect table of contents in document and analyze structure
    
    Args:
        context: Serialized PageIndexContext with pages data
        
    Returns:
        Updated context with toc_info populated
    """
    
    try:
        # Reconstruct context using consistent approach
        context = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context.config.global_config.log_dir) / context.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context.log_step("toc_detector", "started")
        
        # Load pages data
        pages = context.load_pages()
        if not pages:
            raise PageIndexToolError("No pages data available for TOC detection")
        
        # Get configuration
        detector_config = context.config.toc_detector
        model = context.config.global_config.model
        
        # Find TOC pages
        context.log_step("toc_detector", "searching_toc_pages")
        toc_page_list = find_toc_pages(pages, detector_config.toc_check_page_num, model)
        
        if not toc_page_list:
            # No TOC found
            context.toc_info = {
                "found": False,
                "pages": [],
                "content": "",
                "has_page_numbers": False,
                "confidence": 1.0
            }
            context.log_step("toc_detector", "completed", {"toc_found": False})
        else:
            # TOC found, extract content and analyze
            context.log_step("toc_detector", "extracting_toc_content", {"toc_pages": toc_page_list})
            toc_result = extract_toc_content(pages, toc_page_list, model)
            
            context.toc_info = {
                "found": True,
                "pages": toc_page_list,
                "content": toc_result["content"],
                "has_page_numbers": toc_result["has_page_numbers"],
                "confidence": 0.9  # High confidence when TOC is found
            }
            context.log_step("toc_detector", "completed", {
                "toc_found": True,
                "toc_pages": len(toc_page_list),
                "has_page_numbers": toc_result["has_page_numbers"]
            })
        
        # Save checkpoint
        context.save_checkpoint(log_dir)
        
        return {
            "success": True,
            "context": context.to_dict(),
            "confidence": context.toc_info["confidence"],
            "metrics": {
                "toc_found": context.toc_info["found"],
                "toc_pages_count": len(context.toc_info["pages"]),
                "has_page_numbers": context.toc_info["has_page_numbers"]
            },
            "errors": [],
            "suggestions": []
        }
        
    except Exception as e:
        # Handle errors
        if 'context' in locals():
            context.log_step("toc_detector", "failed", {"error": str(e)})
            context.save_checkpoint(log_dir)
            
        suggestions = create_recovery_suggestions("toc_detection", str(e))
        suggestions.extend([
            "Try increasing toc_check_page_num to search more pages",
            "Proceed with no-TOC processing strategy",
            "Manual TOC page specification may be needed"
        ])
        
        return {
            "success": False,
            "context": context.to_dict() if 'context' in locals() else None,
            "metrics": {
                "toc_found": False,
                "toc_pages_count": 0,
                "has_page_numbers": False
            },
            "errors": [str(e)],
            "suggestions": suggestions
        }


def find_toc_pages(pages: List[tuple], max_pages: int, model: str) -> List[int]:
    """Find pages containing table of contents"""
    toc_page_list = []
    last_page_is_yes = False
    
    for i in range(min(max_pages, len(pages))):
        page_text = pages[i][0]
        detected_result = detect_toc_single_page(page_text, model)
        
        if detected_result == 'yes':
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes:
            # Found the end of TOC
            break
    
    return toc_page_list


def detect_toc_single_page(content: str, model: str) -> str:
    """Detect if a single page contains table of contents"""
    client = openai.OpenAI()
    
    prompt = f"""
You are an expert document analyzer tasked with detecting genuine table of contents pages in PDF documents.

Your job is to determine if the given text represents a true TABLE OF CONTENTS that outlines the document's structure.

## CRITERIA FOR TRUE TABLE OF CONTENTS:
A genuine TOC must have ALL of the following characteristics:
1. **Hierarchical organization**: Shows document structure with chapters, sections, and subsections
2. **Descriptive titles**: Contains meaningful section/chapter names that describe document content
3. **Page references**: Includes page numbers, dots/dashes leading to numbers, or similar navigation aids
4. **Structural purpose**: Serves as a navigation guide for the entire document
5. **Content organization**: Lists major topics/themes that would logically organize a full document

## EXPLICIT EXCLUSIONS (These are NOT table of contents):
- **Lists of figures, tables, equations, or illustrations**
- **Mathematical formulas, equations, or technical notation**
- **Bibliographies, references, or citation lists**
- **Index or alphabetical listings**
- **Abstract, summary, or executive summary sections**
- **Notation lists, symbol definitions, or acronym explanations**
- **Appendix listings or supplementary material lists**
- **Glossary or terminology definitions**
- **Author information, acknowledgments, or preface content**
- **Single-level bullet points or simple lists**
- **Code listings, algorithm descriptions, or technical specifications**

## ANALYSIS INSTRUCTIONS:
1. First, identify if there are hierarchical section titles (not just lists)
2. Check for page number references or navigation elements
3. Verify the content describes document organization (not just data/figures)
4. Confirm this serves as a structural guide to a larger work

<GivenText> 
{content}
</GivenText>

Return the following JSON format:
{{
    "thinking": "<Detailed analysis: What hierarchical structure do you see? Are there page numbers? Does this organize document content or just list items? Why does this qualify or disqualify as a TOC?>",
    "toc_detected": "<yes or no>",
    "confidence": "<high/medium/low>",
    "key_indicators": "<List the specific elements that support your decision>"
}}

Directly return the final JSON structure. Do not output anything else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_content = extract_json(response.choices[0].message.content)
        return json_content.get('toc_detected', 'no')
    except Exception as e:
        print(f"Error in TOC detection: {e}")
        return 'no'


def extract_toc_content(pages: List[tuple], toc_page_list: List[int], model: str) -> Dict[str, Any]:
    """Extract TOC content and detect if it has page numbers"""
    import re
    
    # Combine TOC pages
    toc_content = ""
    for page_index in toc_page_list:
        toc_content += pages[page_index][0]
    
    # Transform dots to colons for better parsing
    toc_content = re.sub(r'\.{5,}', ': ', toc_content)
    toc_content = re.sub(r'(?:\. ){5,}\.?', ': ', toc_content)
    
    # Detect if TOC has page numbers
    has_page_numbers = detect_page_numbers_in_toc(toc_content, model)
    
    return {
        "content": toc_content,
        "has_page_numbers": has_page_numbers
    }


def detect_page_numbers_in_toc(toc_content: str, model: str) -> bool:
    """Detect if TOC contains page numbers"""
    client = openai.OpenAI()
    
    prompt = f"""
    You will be given a table of contents.

    Your job is to detect if there are page numbers/indices given within the table of contents.

    Given text: {toc_content}

    Reply format:
    {{
        "thinking": <why do you think there are page numbers/indices given within the table of contents>
        "page_index_given_in_toc": "<yes or no>"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_content = extract_json(response.choices[0].message.content)
        return json_content.get('page_index_given_in_toc', 'no') == 'yes'
    except Exception as e:
        print(f"Error in page number detection: {e}")
        return False
