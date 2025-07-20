import json
import openai
import copy
import math
from pathlib import Path
from typing import Dict, Any, List
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import extract_json, count_tokens, create_recovery_suggestions

def structure_extractor_tool(context: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    """
    Extract document hierarchy using specified strategy
    
    Args:
        context: Serialized PageIndexContext with pages and toc_info
        strategy: "toc_with_pages" | "toc_no_pages" | "no_toc"
        
    Returns:
        Updated context with structure_raw populated
    """
    
    try:
        # Reconstruct context
        context = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context.config["global"]["log_dir"]) / context.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context.log_step("structure_extractor", "started", {"strategy": strategy})
        
        # Load pages data
        pages = context.load_pages()
        if not pages:
            raise PageIndexToolError("No pages data available for structure extraction")
        
        # Get configuration
        extractor_config = context.config["structure_extractor"]
        model = context.config["global"]["model"]
        
        # Execute strategy-specific extraction
        if strategy == "toc_with_pages":
            if not context.toc_info.get("found") or not context.toc_info.get("has_page_numbers"):
                raise PageIndexToolError("Strategy 'toc_with_pages' requires TOC with page numbers")
            
            structure_raw = extract_with_toc_pages(
                pages, context.toc_info, extractor_config, model, context
            )
            confidence = 0.9
            
        elif strategy == "toc_no_pages":
            if not context.toc_info.get("found"):
                raise PageIndexToolError("Strategy 'toc_no_pages' requires TOC to be found")
            
            structure_raw = extract_with_toc_no_pages(
                pages, context.toc_info, extractor_config, model, context
            )
            confidence = 0.7
            
        elif strategy == "no_toc":
            structure_raw = extract_without_toc(
                pages, extractor_config, model, context
            )
            confidence = 0.6
            
        else:
            raise PageIndexToolError(f"Unknown extraction strategy: {strategy}")
        
        # Validate extraction results
        if not structure_raw:
            raise PageIndexToolError("No structure extracted from document")
        
        # Store results
        context.structure_raw = structure_raw
        
        # Log success
        context.log_step("structure_extractor", "completed", {
            "strategy": strategy,
            "items_extracted": len(structure_raw),
            "confidence": confidence
        })
        
        # Save checkpoint
        context.save_checkpoint(log_dir)
        
        return {
            "success": True,
            "context": context.to_dict(),
            "confidence": confidence,
            "metrics": {
                "strategy_used": strategy,
                "items_extracted": len(structure_raw),
                "has_page_numbers": any('physical_index' in item for item in structure_raw)
            },
            "errors": [],
            "suggestions": []
        }
        
    except PageIndexToolError as e:
        # Handle tool-specific errors
        if 'context' in locals():
            context.log_step("structure_extractor", "failed", {
                "error": str(e),
                "strategy": strategy
            })
            context.save_checkpoint(log_dir)
        
        suggestions = create_recovery_suggestions("structure_extraction", str(e))
        
        # Add strategy-specific suggestions
        if strategy == "toc_with_pages":
            suggestions.extend([
                "Try 'toc_no_pages' strategy instead",
                "Verify TOC actually contains page numbers"
            ])
        elif strategy == "toc_no_pages":
            suggestions.extend([
                "Try 'no_toc' strategy instead",
                "Check if TOC was properly detected"
            ])
        elif strategy == "no_toc":
            suggestions.extend([
                "Document may lack clear section structure",
                "Try adjusting max_token_num_each_node"
            ])
        
        return {
            "success": False,
            "context": context.to_dict() if 'context' in locals() else context,
            "confidence": 0.0,
            "metrics": {},
            "errors": [str(e)],
            "suggestions": suggestions
        }
        
    except Exception as e:
        # Handle unexpected errors
        if 'context' in locals():
            context.log_step("structure_extractor", "failed", {
                "error": str(e),
                "error_type": "unexpected",
                "strategy": strategy
            })
            context.save_checkpoint(log_dir)
        
        suggestions = create_recovery_suggestions("structure_extraction", str(e))
        
        return {
            "success": False,
            "context": context.to_dict() if 'context' in locals() else context,
            "confidence": 0.0,
            "metrics": {},
            "errors": [str(e)],
            "suggestions": suggestions
        }


def extract_with_toc_pages(pages: List[tuple], toc_info: Dict[str, Any], 
                          config: Dict[str, Any], model: str, context) -> List[Dict[str, Any]]:
    """Extract structure using TOC with page numbers"""
    
    context.log_step("structure_extractor", "transforming_toc")
    
    # Transform TOC content to structured format
    toc_structured = transform_toc_to_json(toc_info["content"], model)
    
    # Convert page numbers to integers
    toc_structured = convert_page_to_int(toc_structured)
    
    context.log_step("structure_extractor", "matching_physical_indices")
    
    # Create sample content for physical index matching
    toc_pages = toc_info["pages"]
    start_page_index = toc_pages[-1] + 1 if toc_pages else 0
    max_check_pages = min(20, len(pages) - start_page_index)
    
    sample_content = ""
    for page_idx in range(start_page_index, start_page_index + max_check_pages):
        if page_idx < len(pages):
            sample_content += f"<physical_index_{page_idx+1}>\n{pages[page_idx][0]}\n<physical_index_{page_idx+1}>\n\n"
    
    # Extract physical indices for some TOC items
    toc_with_physical = extract_toc_physical_indices(toc_structured, sample_content, model)
    
    # Calculate page offset
    offset = calculate_page_offset(toc_structured, toc_with_physical, start_page_index + 1)
    
    # Apply offset to all TOC items
    final_structure = apply_page_offset(toc_structured, offset)
    
    return final_structure


def extract_with_toc_no_pages(pages: List[tuple], toc_info: Dict[str, Any],
                             config: Dict[str, Any], model: str, context) -> List[Dict[str, Any]]:
    """Extract structure using TOC without page numbers"""
    
    context.log_step("structure_extractor", "transforming_toc")
    
    # Transform TOC content to structured format
    toc_structured = transform_toc_to_json(toc_info["content"], model)
    
    context.log_step("structure_extractor", "matching_content_to_structure")
    
    # Create content with page markers
    page_contents = []
    token_lengths = []
    for page_index in range(len(pages)):
        page_text = f"<physical_index_{page_index + 1}>\n{pages[page_index][0]}\n<physical_index_{page_index + 1}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    # Group pages to manage token limits
    group_texts = page_list_to_group_text(page_contents, token_lengths, config["max_token_num_each_node"])
    
    # Match TOC items to content groups
    toc_with_indices = copy.deepcopy(toc_structured)
    for group_text in group_texts:
        toc_with_indices = match_toc_to_content(group_text, toc_with_indices, model)
    
    # Convert physical indices to integers
    final_structure = convert_physical_index_to_int(toc_with_indices)
    
    return final_structure


def extract_without_toc(pages: List[tuple], config: Dict[str, Any], 
                       model: str, context) -> List[Dict[str, Any]]:
    """Extract structure without TOC by analyzing content"""
    
    context.log_step("structure_extractor", "analyzing_content_structure")
    
    # Create content with page markers
    page_contents = []
    token_lengths = []
    for page_index in range(len(pages)):
        page_text = f"<physical_index_{page_index + 1}>\n{pages[page_index][0]}\n<physical_index_{page_index + 1}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, model))
    
    # Group pages to manage token limits
    group_texts = page_list_to_group_text(page_contents, token_lengths, config["max_token_num_each_node"])
    
    context.log_step("structure_extractor", "generating_structure", {"groups": len(group_texts)})
    
    # Generate structure from first group
    structure = generate_structure_from_content(group_texts[0], model)
    
    # Extend structure with remaining groups
    for group_text in group_texts[1:]:
        additional_structure = generate_additional_structure(structure, group_text, model)
        structure.extend(additional_structure)
    
    # Convert physical indices to integers
    final_structure = convert_physical_index_to_int(structure)
    
    return final_structure


def transform_toc_to_json(toc_content: str, model: str) -> List[Dict[str, Any]]:
    """Transform raw TOC content to structured JSON format"""
    client = openai.OpenAI()
    
    prompt = f"""
Transform this table of contents into JSON format.

RULES:
1. Extract hierarchical structure (1, 1.1, 1.2, 2, 2.1, etc.) - use None if no numbering
2. Keep original titles, fix only spacing/formatting issues
3. Extract page numbers if present - use None if missing
4. Handle multi-column layouts and varied indentation
5. Ignore headers, footers, and non-content elements

OUTPUT FORMAT:
{{
"table_of_contents": [
  {{"structure": "1", "title": "Introduction", "page": 5}},
  {{"structure": "1.1", "title": "Overview", "page": 6}},
  {{"structure": None, "title": "Untitled Section", "page": None}}
]
}}

Return only valid JSON, no explanations.

TOC Content:
{toc_content}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_content = extract_json(response.choices[0].message.content)
        return json_content.get('table_of_contents', [])
    except Exception as e:
        print(f"Error in TOC transformation: {e}")
        return []


def extract_toc_physical_indices(toc_structured: List[Dict[str, Any]], 
                                content: str, model: str) -> List[Dict[str, Any]]:
    """Extract physical indices for TOC items from document content"""
    client = openai.OpenAI()
    
    prompt = f"""
Match TOC sections to physical page locations.

TASK: Find where each TOC section starts in the document pages.

RULES:
1. Look for section titles in the document content
2. Match the physical_index tag where the section begins
3. Only add physical_index if section is clearly found
4. Keep exact format: "<physical_index_X>"
5. Skip sections not found in provided pages

OUTPUT: Return the TOC JSON with physical_index added where found.

TOC:
{json.dumps(toc_structured, indent=2)}

Document Pages:
{content}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in physical index extraction: {e}")
        return []


def match_toc_to_content(content: str, toc_items: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Match TOC items to content and add physical indices"""
    client = openai.OpenAI()
    
    prompt = f"""
Update TOC items with physical_index where sections start in this document part.

TASK: Find TOC section titles that begin in the current document content.

RULES:
1. Look for exact or close title matches in the content
2. If section starts here, add "physical_index": "<physical_index_X>"
3. If section doesn't start here, leave physical_index unchanged
4. Preserve all existing data from previous processing
5. Match section beginnings, not just mentions

OUTPUT: Return updated TOC JSON with new physical_index values added.

Document Content:
{content}

Current TOC Structure:
{json.dumps(toc_items, indent=2)}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in content matching: {e}")
        return toc_items


def generate_structure_from_content(content: str, model: str) -> List[Dict[str, Any]]:
    """Generate initial structure from document content"""
    client = openai.OpenAI()
    
    prompt = f"""
Extract document structure from content.

TASK: Identify sections, subsections, and their hierarchy.

RULES:
1. Detect headings, titles, and section breaks
2. Assign hierarchical numbers (1, 1.1, 1.2, 2, 2.1, etc.)
3. Use original titles, fix only spacing issues
4. Find physical_index where each section starts: "<physical_index_X>"
5. Include all significant structural elements
6. Skip headers, footers, page numbers

OUTPUT FORMAT:
[
  {{"structure": "1", "title": "Introduction", "physical_index": "<physical_index_1>"}},
  {{"structure": "1.1", "title": "Overview", "physical_index": "<physical_index_2>"}}
]

Document Content:
{content}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in structure generation: {e}")
        return []


def generate_additional_structure(existing_structure: List[Dict[str, Any]], 
                                 content: str, model: str) -> List[Dict[str, Any]]:
    """Generate additional structure from content to extend existing structure"""
    client = openai.OpenAI()
    
    prompt = f"""
Extract NEW sections from content to extend existing structure.

TASK: Find sections in current content that continue the document structure.

RULES:
1. Continue numbering from existing structure (check last section number)
2. Only return NEW sections found in current content
3. Maintain hierarchical consistency with existing structure
4. Use original titles, fix only spacing
5. Find physical_index where each NEW section starts
6. Don't duplicate existing sections

EXISTING STRUCTURE (last items):
{json.dumps(existing_structure[-3:] if len(existing_structure) > 3 else existing_structure, indent=2)}

OUTPUT: Return only NEW sections as JSON array.

Current Content:
{content}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return extract_json(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in additional structure generation: {e}")
        return []


# Utility functions
def convert_page_to_int(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert page numbers from string to int"""
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                pass
    return data


def convert_physical_index_to_int(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert physical index from string format to int"""
    for item in data:
        if 'physical_index' in item and isinstance(item['physical_index'], str):
            if item['physical_index'].startswith('<physical_index_'):
                try:
                    item['physical_index'] = int(item['physical_index'].split('_')[-1].rstrip('>').strip())
                except ValueError:
                    item['physical_index'] = None
    return data


def page_list_to_group_text(page_contents: List[str], token_lengths: List[int], 
                           max_tokens: int = 20000, overlap_page: int = 1) -> List[str]:
    """Group pages into text chunks respecting token limits"""
    num_tokens = sum(token_lengths)
    
    if num_tokens <= max_tokens:
        return ["".join(page_contents)]
    
    subsets = []
    current_subset = []
    current_token_count = 0
    
    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)
    
    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:
            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])
        
        current_subset.append(page_content)
        current_token_count += page_tokens
    
    if current_subset:
        subsets.append(''.join(current_subset))
    
    return subsets


def calculate_page_offset(toc_structured: List[Dict[str, Any]], 
                         toc_with_physical: List[Dict[str, Any]], 
                         start_page_index: int) -> int:
    """Calculate offset between TOC page numbers and physical indices"""
    differences = []
    
    for toc_item in toc_structured:
        for phys_item in toc_with_physical:
            if (toc_item.get('title') == phys_item.get('title') and 
                'page' in toc_item and 'physical_index' in phys_item):
                try:
                    physical_index = int(phys_item['physical_index'].split('_')[-1].rstrip('>'))
                    page_number = toc_item['page']
                    if isinstance(page_number, int) and physical_index >= start_page_index:
                        difference = physical_index - page_number
                        differences.append(difference)
                except (ValueError, AttributeError):
                    continue
    
    if not differences:
        return 0
    
    # Return most common difference
    from collections import Counter
    return Counter(differences).most_common(1)[0][0]


def apply_page_offset(toc_structured: List[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    """Apply page offset to convert page numbers to physical indices"""
    for item in toc_structured:
        if 'page' in item and item['page'] is not None:
            item['physical_index'] = item['page'] + offset
            del item['page']
    return toc_structured
