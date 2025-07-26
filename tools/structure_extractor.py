from typing import Dict, Any, List, Tuple, Optional
import json
import openai
import copy
import math
from pathlib import Path
from core.config import PageIndexConfig
from typing import Dict, Any, List
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import extract_json, count_tokens, create_recovery_suggestions
from core.llm_batch_utils import LLMBatcher, BatchItem

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
        # Reconstruct context using consistent approach
        context = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context.config.global_config.log_dir) / context.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context.log_step("structure_extractor", "started", {"strategy": strategy})
        
        # Load pages data
        pages = context.load_pages()
        if not pages:
            raise PageIndexToolError("No pages data available for structure extraction")
        
        # Get configuration
        extractor_config = context.config.structure_extractor
        model = context.config.global_config.model
        
        # Execute strategy-specific extraction with resilient fallback
        attempted_strategy = strategy
        fallback_attempted = False
        structure_raw = None
        confidence = 0.0
        
        if strategy == "toc_with_pages":
            if not context.toc_info.get("found") or not context.toc_info.get("has_page_numbers"):
                context.log_step("structure_extractor", "strategy_fallback", {
                    "original_strategy": strategy,
                    "reason": "TOC not found or lacks page numbers",
                    "toc_found": context.toc_info.get("found", False),
                    "has_page_numbers": context.toc_info.get("has_page_numbers", False),
                    "fallback_to": "no_toc"
                })
                # Automatic fallback to no_toc strategy
                strategy = "no_toc"
                fallback_attempted = True
            else:
                structure_raw = extract_with_toc_pages(
                    pages, context.toc_info, extractor_config, model, context
                )
                confidence = 0.9
            
        elif strategy == "toc_no_pages":
            if not context.toc_info.get("found"):
                context.log_step("structure_extractor", "strategy_fallback", {
                    "original_strategy": strategy,
                    "reason": "TOC not found",
                    "toc_found": context.toc_info.get("found", False),
                    "fallback_to": "no_toc"
                })
                # Automatic fallback to no_toc strategy
                strategy = "no_toc"
                fallback_attempted = True
            else:
                structure_raw = extract_with_toc_no_pages(
                    pages, context.toc_info, extractor_config, model, context
                )
                confidence = 0.7
        
        # Execute fallback or direct no_toc strategy
        if strategy == "no_toc":
            structure_raw = extract_without_toc(
                pages, extractor_config, model, context
            )
            confidence = 0.6 if not fallback_attempted else 0.5  # Lower confidence for fallback
        
        # Validate that we have a valid strategy and results
        if structure_raw is None:
            raise PageIndexToolError(f"Unknown extraction strategy: {attempted_strategy}")
        
        # Validate extraction results
        if not structure_raw:
            raise PageIndexToolError("No structure extracted from document")
        
        # Store results
        context.structure_raw = structure_raw
        
        # Log completion with strategy used and fallback information
        context.log_step("structure_extractor", "completed", {
            "strategy": strategy,
            "original_strategy": attempted_strategy,
            "fallback_attempted": fallback_attempted,
            "confidence": confidence,
            "structure_count": len(structure_raw),
            "toc_state": {
                "found": context.toc_info.get("found", False),
                "has_page_numbers": context.toc_info.get("has_page_numbers", False),
                "pages_count": len(context.toc_info.get("pages", []))
            }
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
                          config: PageIndexConfig, model: str, context) -> List[Dict[str, Any]]:
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
                             config: PageIndexConfig, model: str, context) -> List[Dict[str, Any]]:
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
    group_texts = page_list_to_group_text(page_contents, token_lengths, config.max_token_num_each_node)

    # Match TOC items to content groups using batch processing
    toc_with_indices = copy.deepcopy(toc_structured)
    
    if len(group_texts) > 1:
        # Use batch processing for multiple groups
        try:
            from core.async_utils import run_async_safe
            toc_with_indices = run_async_safe(batch_match_toc_to_content(group_texts, toc_with_indices, model))
            context.log_step("structure_extractor", "batch_toc_matching_success", {"groups": len(group_texts)})
        except Exception as e:
            context.log_step("structure_extractor", "batch_toc_matching_failed", {"error": str(e)})
            # Fallback to individual processing
            for group_text in group_texts:
                toc_with_indices = match_toc_to_content(group_text, toc_with_indices, model)
    else:
        # Single group - use individual processing
        for group_text in group_texts:
            toc_with_indices = match_toc_to_content(group_text, toc_with_indices, model)
    
    # Convert physical indices to integers
    final_structure = convert_physical_index_to_int(toc_with_indices)
    
    return final_structure


def extract_without_toc(pages: List[tuple], config: PageIndexConfig, 
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
    group_texts = page_list_to_group_text(page_contents, token_lengths, config.max_token_num_each_node)
    
    context.log_step("structure_extractor", "generating_structure", {"groups": len(group_texts)})
    
    # Generate structure using batch processing when possible
    if len(group_texts) > 1:
        # Use batch processing for multiple groups
        try:
            from core.async_utils import run_async_safe
            structure = run_async_safe(batch_generate_structure_from_content(group_texts, model))
            context.log_step("structure_extractor", "batch_structure_generation_success", {"groups": len(group_texts)})
        except Exception as e:
            context.log_step("structure_extractor", "batch_structure_generation_failed", {"error": str(e)})
            # Fallback to individual processing
            structure = generate_structure_from_content(group_texts[0], model)
            for group_text in group_texts[1:]:
                additional_structure = generate_additional_structure(structure, group_text, model)
                structure.extend(additional_structure)
    else:
        # Single group - use individual processing
        structure = generate_structure_from_content(group_texts[0], model)
    
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


async def batch_transform_toc_to_json(toc_contents: List[str], model: str) -> List[List[Dict[str, Any]]]:
    """Token-aware batching for transforming multiple TOC contents to JSON format"""
    if not toc_contents:
        return []
    
    # Create batch items for each TOC content
    batch_items = []
    for i, toc_content in enumerate(toc_contents):
        # Check token count before adding to batch
        content_tokens = count_tokens(toc_content)
        if content_tokens > 50000:  # Conservative limit for TOC content
            print(f"Warning: TOC content {i} too large ({content_tokens} tokens), processing individually")
            # Process large TOC individually
            result = transform_toc_to_json(toc_content, model)
            continue
            
        batch_items.append(BatchItem(
            id=f"toc_{i}",
            content=toc_content,
            metadata={"toc_index": i, "token_count": content_tokens}
        ))
    
    if not batch_items:
        # All TOCs were too large, process individually
        results = []
        for toc_content in toc_contents:
            result = transform_toc_to_json(toc_content, model)
            results.append(result)
        return results
    
    # Use token-aware batching
    batcher = LLMBatcher(model, max_tokens=120000)
    batch_results = await batcher.batch_toc_operations(batch_items, "transform_toc")
    
    # Process results and maintain order
    results = [[] for _ in toc_contents]  # Initialize with empty lists
    
    for result in batch_results:
        if not result.error:
            try:
                toc_data = json.loads(result.result)
                toc_index = int(result.id.split('_')[1])  # Extract index from "toc_X"
                results[toc_index] = toc_data.get('table_of_contents', [])
            except Exception as e:
                print(f"Error processing batch result for {result.id}: {e}")
                # Fallback to individual processing
                toc_index = int(result.id.split('_')[1])
                results[toc_index] = transform_toc_to_json(toc_contents[toc_index], model)
        else:
            print(f"Batch error for {result.id}: {result.error}")
            # Fallback to individual processing
            toc_index = int(result.id.split('_')[1])
            results[toc_index] = transform_toc_to_json(toc_contents[toc_index], model)
    
    return results


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
    # For single content chunk, use original approach (no batching benefit)
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


async def batch_match_toc_to_content(content_chunks: List[str], toc_items: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Token-aware batching for matching TOC items across multiple content chunks with proper attribute preservation"""
    if not content_chunks:
        return toc_items
    
    # For single chunk, use individual processing (no batching benefit)
    if len(content_chunks) == 1:
        return match_toc_to_content(content_chunks[0], toc_items, model)
    
    # Create batch items for each content chunk with current TOC state
    batch_items = []
    current_toc_state = copy.deepcopy(toc_items)
    
    for i, content in enumerate(content_chunks):
        # Check token count to prevent overflow
        content_tokens = count_tokens(content)
        toc_tokens = count_tokens(json.dumps(current_toc_state, indent=2))
        total_tokens = content_tokens + toc_tokens + 500  # Buffer for prompt
        
        if total_tokens > 100000:  # Conservative limit
            print(f"Warning: Content chunk {i} too large ({total_tokens} tokens), using individual processing")
            # Process this chunk individually and update state
            current_toc_state = match_toc_to_content(content, current_toc_state, model)
            continue
            
        batch_items.append(BatchItem(
            id=f"chunk_{i}",
            content=f"Document Content:\n{content}\n\nCurrent TOC Structure:\n{json.dumps(current_toc_state, indent=2)}",
            metadata={"chunk_index": i, "toc_snapshot": copy.deepcopy(current_toc_state)}
        ))
    
    if not batch_items:
        # All chunks were too large, already processed individually
        return current_toc_state
    
    try:
        # Use token-aware batching
        batcher = LLMBatcher(model, max_tokens=120000)
        results = await batcher.batch_toc_operations(batch_items, "match_content")
        
        # Process results with careful attribute preservation
        final_toc = copy.deepcopy(current_toc_state)
        
        for result in results:
            if not result.error and result.result.strip():
                try:
                    # Parse the batch result
                    chunk_result = json.loads(result.result)
                    
                    # Validate that result is a list of TOC items
                    if not isinstance(chunk_result, list):
                        raise ValueError(f"Expected list, got {type(chunk_result)}")
                    
                    # Carefully merge only the physical_index updates
                    for updated_item in chunk_result:
                        if not isinstance(updated_item, dict) or 'title' not in updated_item:
                            continue
                            
                        # Find matching item in final_toc by title
                        for j, toc_item in enumerate(final_toc):
                            if toc_item.get('title') == updated_item.get('title'):
                                # Only update physical_index if it's new and valid
                                if ('physical_index' in updated_item and 
                                    updated_item['physical_index'] and 
                                    'physical_index' not in toc_item):
                                    final_toc[j]['physical_index'] = updated_item['physical_index']
                                break
                                
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error parsing batch result for {result.id}: {e}")
                    # Fallback: process this chunk individually
                    chunk_index = result.id.split('_')[1]
                    if chunk_index.isdigit():
                        chunk_idx = int(chunk_index)
                        if chunk_idx < len(content_chunks):
                            final_toc = match_toc_to_content(content_chunks[chunk_idx], final_toc, model)
            else:
                # Batch failed for this chunk, process individually
                print(f"Batch failed for {result.id}: {result.error}")
                chunk_index = result.id.split('_')[1]
                if chunk_index.isdigit():
                    chunk_idx = int(chunk_index)
                    if chunk_idx < len(content_chunks):
                        final_toc = match_toc_to_content(content_chunks[chunk_idx], final_toc, model)
        
        return final_toc
        
    except Exception as e:
        print(f"Batch processing failed completely: {e}")
        # Complete fallback to individual processing
        final_toc = copy.deepcopy(toc_items)
        for content in content_chunks:
            final_toc = match_toc_to_content(content, final_toc, model)
        return final_toc


def generate_structure_from_content(content: str, model: str) -> List[Dict[str, Any]]:
    """Generate initial structure from document content"""
    # For single content chunk, use original approach
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


async def batch_generate_structure_from_content(content_chunks: List[str], model: str) -> List[Dict[str, Any]]:
    """Token-aware batching for generating structure from multiple content chunks with proper sequential processing"""
    if not content_chunks:
        return []
    
    # For single chunk, use individual processing (no batching benefit)
    if len(content_chunks) == 1:
        return generate_structure_from_content(content_chunks[0], model)
    
    # IMPORTANT: Structure generation requires sequential processing to maintain
    # proper hierarchical numbering and continuity across chunks.
    # Batch processing would break the sequential dependency where each chunk
    # extends the structure from previous chunks.
    
    # Use the original sequential logic to maintain functionality
    try:
        # Generate structure from first group
        structure = generate_structure_from_content(content_chunks[0], model)
        
        # Extend structure with remaining groups sequentially
        for i, group_text in enumerate(content_chunks[1:], 1):
            try:
                additional_structure = generate_additional_structure(structure, group_text, model)
                structure.extend(additional_structure)
            except Exception as e:
                print(f"Error processing content chunk {i}: {e}")
                # Continue with remaining chunks even if one fails
                continue
        
        return structure
        
    except Exception as e:
        print(f"Error in structure generation: {e}")
        # Complete fallback - process each chunk independently and merge
        all_structure = []
        for i, content in enumerate(content_chunks):
            try:
                chunk_structure = generate_structure_from_content(content, model)
                all_structure.extend(chunk_structure)
            except Exception as chunk_error:
                print(f"Error processing chunk {i}: {chunk_error}")
                continue
        
        return all_structure


async def batch_generate_structure_from_content_experimental(content_chunks: List[str], model: str) -> List[Dict[str, Any]]:
    """EXPERIMENTAL: True batch processing for structure generation (may break sequential dependencies)"""
    if not content_chunks:
        return []
    
    # Create batch items for each content chunk
    batch_items = []
    for i, content in enumerate(content_chunks):
        # Check token count before adding to batch
        content_tokens = count_tokens(content)
        if content_tokens > 80000:  # Conservative limit for structure generation
            print(f"Warning: Content chunk {i} too large ({content_tokens} tokens), processing individually")
            # Process large chunk individually and add to results later
            continue
            
        batch_items.append(BatchItem(
            id=f"content_chunk_{i}",
            content=content,
            metadata={"chunk_index": i, "token_count": content_tokens}
        ))
    
    if not batch_items:
        # All chunks were too large, fall back to sequential processing
        return await batch_generate_structure_from_content(content_chunks, model)
    
    try:
        # Use token-aware batching
        batcher = LLMBatcher(model, max_tokens=100000)  # Conservative limit
        
        # Define the operation prompt template
        base_prompt = """
Extract document structure from the following content chunks independently.

TASK: Identify sections, subsections, and their hierarchy for each chunk.

RULES:
1. Detect headings, titles, and section breaks
2. Assign hierarchical numbers starting from 1 for each chunk
3. Use original titles, fix only spacing issues
4. Find physical_index where each section starts: "<physical_index_X>"
5. Include all significant structural elements
6. Skip headers, footers, page numbers

OUTPUT FORMAT:
{
  "results": [
    {"id": "content_chunk_0", "result": [{"structure": "1", "title": "Introduction", "physical_index": "<physical_index_1>"}]},
    {"id": "content_chunk_1", "result": [{"structure": "1", "title": "Methods", "physical_index": "<physical_index_5>"}]}
  ]
}

"""
        
        # Split into token-aware batches
        batches = batcher._split_items_by_token_limit(batch_items, base_prompt)
        
        all_structure = []
        for batch in batches:
            try:
                # Build batch prompt
                batch_prompt = base_prompt
                for item in batch:
                    batch_prompt += f"ID: {item.id}\n"
                    batch_prompt += f"Document Content:\n{item.content}\n\n"
                
                response = await batcher.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0,
                )
                
                # Parse batch response
                try:
                    response_data = json.loads(response.choices[0].message.content)
                    results = response_data.get("results", [])
                    
                    # Sort results by chunk index to maintain order
                    sorted_results = sorted(results, key=lambda x: int(x.get("id", "content_chunk_0").split("_")[-1]))
                    
                    for result in sorted_results:
                        structure_items = result.get("result", [])
                        all_structure.extend(structure_items)
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing batch response: {e}")
                    # Fallback to individual processing for this batch
                    for item in batch:
                        chunk_index = item.metadata["chunk_index"]
                        if chunk_index < len(content_chunks):
                            structure = generate_structure_from_content(content_chunks[chunk_index], model)
                            all_structure.extend(structure)
                        
            except Exception as e:
                print(f"Error in batch structure generation: {e}")
                # Fallback to individual processing for this batch
                for item in batch:
                    chunk_index = item.metadata["chunk_index"]
                    if chunk_index < len(content_chunks):
                        structure = generate_structure_from_content(content_chunks[chunk_index], model)
                        all_structure.extend(structure)
        
        return all_structure
        
    except Exception as e:
        print(f"Experimental batch processing failed: {e}")
        # Fallback to sequential processing
        return await batch_generate_structure_from_content(content_chunks, model)


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
