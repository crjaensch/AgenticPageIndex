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
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    {{
    "table_of_contents": [
        {{
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        }},
        ...
        ],
    }}
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else.
    
    Given table of contents:
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
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format: 
    [
        {{
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        }},
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the section is not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else.
    
    Table of contents:
    {json.dumps(toc_structured, indent=2)}
    
    Document pages:
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
    You are given an JSON structure of a document and a partial part of the document. Your task is to check if the title that is described in the structure is started in the partial given document.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X. 

    If the full target section starts in the partial given document, insert the given JSON structure with the "start": "yes", and "start_index": "<physical_index_X>".

    If the full target section does not start in the partial given document, insert "start": "no",  "start_index": None.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x" or None> (string),
                "title": <title of the section>,
                "start": "<yes or no>",
                "physical_index": "<physical_index_X> (keep the format)" or None
            }},
            ...
        ]    
    The given structure contains the result of the previous part, you need to fill the result of the current part, do not change the previous result.
    Directly return the final JSON structure. Do not output anything else.
    
    Current Partial Document:
    {content}
    
    Given Structure:
    {json.dumps(toc_items, indent=2)}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_result = extract_json(response.choices[0].message.content)
        
        # Remove temporary 'start' field
        for item in json_result:
            if 'start' in item:
                del item['start']
        
        return json_result
    except Exception as e:
        print(f"Error in content matching: {e}")
        return toc_items


def generate_structure_from_content(content: str, model: str) -> List[Dict[str, Any]]:
    """Generate initial structure from document content"""
    client = openai.OpenAI()
    
    prompt = f"""
    You are an expert in extracting hierarchical tree structure, your task is to generate the tree structure of the document.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            ...
        ]    

    Directly return the final JSON structure. Do not output anything else.
    
    Given text:
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
    You are an expert in extracting hierarchical tree structure.
    You are given a tree structure of the previous part and the text of the current part.
    Your task is to continue the tree structure from the previous part to include the current part.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    For the title, you need to extract the original title from the text, only fix the space inconsistency.

    The provided text contains tags like <physical_index_X> and <physical_index_X> to indicate the start and end of page X. 

    For the physical_index, you need to extract the physical index of the start of the section from the text. Keep the <physical_index_X> format.

    The response should be in the following format. 
        [
            {{
                "structure": <structure index, "x.x.x"> (string),
                "title": <title of the section, keep the original title>,
                "physical_index": "<physical_index_X> (keep the format)"
            }},
            ...
        ]    

    Directly return the additional part of the final JSON structure. Do not output anything else.
    
    Given text:
    {content}
    
    Previous tree structure:
    {json.dumps(existing_structure, indent=2)}"""
    
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
