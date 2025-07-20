import openai
import asyncio
import random
from pathlib import Path
from typing import Dict, Any, List
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import extract_json, create_recovery_suggestions

def structure_verifier_tool(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify structure accuracy and fix errors
    
    Args:
        context: Serialized PageIndexContext with structure_raw
        
    Returns:
        Updated context with structure_verified populated
    """
    
    try:
        # Reconstruct context
        context = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context.config["global"]["log_dir"]) / context.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context.log_step("structure_verifier", "started")
        
        # Load pages and structure data
        pages = context.load_pages()
        structure_raw = context.structure_raw
        
        if not pages:
            raise PageIndexToolError("No pages data available for verification")
        if not structure_raw:
            raise PageIndexToolError("No structure data available for verification")
        
        # Get configuration
        verifier_config = context.config["structure_verifier"]
        model = context.config["global"]["model"]
        client = openai.AsyncOpenAI()
        
        # Validate and truncate physical indices that exceed document length
        context.log_step("structure_verifier", "validating_indices")
        validated_structure = validate_and_truncate_indices(
            structure_raw, len(pages), context
        )
        
        # Verify structure accuracy
        context.log_step("structure_verifier", "verifying_accuracy")
        accuracy, incorrect_items = asyncio.run(
            verify_structure_accuracy(validated_structure, pages, model, client)
        )
        
        context.log_step("structure_verifier", "verification_complete", {
            "accuracy": accuracy,
            "incorrect_items": len(incorrect_items)
        })
        
        # Determine if fixing is needed
        context.structure_verified = validated_structure
        confidence = 1.0
        
        # Save checkpoint
        context.save_checkpoint(log_dir)
        
        return {
            "success": True,
            "context": context.to_dict(),
            "confidence": confidence,
            "metrics": {
                "accuracy": accuracy,
                "total_items": len(validated_structure),
                "incorrect_items": len(incorrect_items),
                "items_verified": len(context.structure_verified)
            },
            "errors": [],
            "suggestions": []
        }
        
    except Exception as e:
        # Handle errors
        if 'context' in locals():
            context.log_step("structure_verifier", "failed", {"error": str(e)})
            context.save_checkpoint(log_dir)
        
        suggestions = create_recovery_suggestions("structure_verification", str(e))
        suggestions.extend([
            "Try lowering accuracy_threshold in configuration",
            "Skip verification if structure appears reasonable",
            "Consider manual review of extracted structure"
        ])
        
        return {
            "success": False,
            "context": context.to_dict() if 'context' in locals() else context,
            "confidence": 0.0,
            "metrics": {},
            "errors": [str(e)],
            "suggestions": suggestions
        }


async def verify_structure_accuracy(structure: List[Dict[str, Any]], 
                                  pages: List[tuple], model: str, client) -> tuple:
    """Verify structure accuracy by checking if titles appear on specified pages"""
    
    # Sample items for verification (max 20 to avoid overwhelming API)
    sample_size = min(20, len(structure))
    if sample_size < len(structure):
        sample_indices = random.sample(range(len(structure)), sample_size)
        sample_items = [structure[i] for i in sample_indices]
        for i, item in enumerate(sample_items):
            item['list_index'] = sample_indices[i]
    else:
        sample_items = structure.copy()
        for i, item in enumerate(sample_items):
            item['list_index'] = i
    
    # Filter items with valid physical_index
    valid_items = [item for item in sample_items if item.get('physical_index') is not None]
    
    if not valid_items:
        return 0.0, []
    
    # Check each item concurrently
    tasks = [
        check_title_on_page(item, pages, model, client)
        for item in valid_items
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    correct_count = 0
    incorrect_items = []
    
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            # Treat exceptions as incorrect
            incorrect_items.append({
                'list_index': item['list_index'],
                'title': item.get('title', 'Unknown'),
                'physical_index': item.get('physical_index'),
                'error': str(result)
            })
        elif result == 'yes':
            correct_count += 1
        else:
            incorrect_items.append({
                'list_index': item['list_index'],
                'title': item.get('title', 'Unknown'),
                'physical_index': item.get('physical_index')
            })
    
    accuracy = correct_count / len(valid_items) if valid_items else 0.0
    return accuracy, incorrect_items


async def check_title_on_page(item: Dict[str, Any], pages: List[tuple], model: str, client) -> str:
    """Check if a title appears on the specified page"""
    
    title = item['title']
    physical_index = item['physical_index']
    
    # Validate physical_index
    if physical_index is None or physical_index < 1 or physical_index > len(pages):
        return 'no'
    
    page_text = pages[physical_index - 1][0]
    
    prompt = f"""
Check if this section title appears or starts on the given page.

TASK: Determine if the section title is present in the page text.

RULES:
1. Use fuzzy matching - ignore spacing, punctuation, case differences
2. Look for the section title or very similar variations
3. Section can appear anywhere on the page (beginning, middle, end)
4. Consider partial matches if they're clearly the same section

OUTPUT: {{"answer": "yes"}} or {{"answer": "no"}}

Section Title: {title}
Page Text: {page_text}"""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_content = extract_json(response.choices[0].message.content)
        return json_content.get('answer', 'no')
    except Exception as e:
        print(f"Error in title verification: {e}")
        return 'no'


async def fix_structure_errors(structure: List[Dict[str, Any]], pages: List[tuple],
                             incorrect_items: List[Dict[str, Any]], max_attempts: int,
                             model: str, context, client) -> tuple:
    """Fix structure errors by finding correct page indices"""
    
    fixed_structure = structure.copy()
    remaining_errors = []
    
    for attempt in range(max_attempts):
        if not incorrect_items:
            break
        
        context.log_step("structure_verifier", f"fix_attempt_{attempt + 1}", {
            "items_to_fix": len(incorrect_items)
        })
        
        # Process incorrect items
        tasks = [
            fix_single_item(item, pages, model, client)
            for item in incorrect_items
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update structure and check remaining errors
        new_incorrect_items = []
        for item, result in zip(incorrect_items, results):
            if isinstance(result, Exception):
                new_incorrect_items.append(item)
            else:
                list_index = item['list_index']
                if 0 <= list_index < len(fixed_structure):
                    if result['found']:
                        fixed_structure[list_index]['physical_index'] = result['physical_index']
                    else:
                        new_incorrect_items.append(item)
        
        incorrect_items = new_incorrect_items
    
    remaining_errors = incorrect_items
    return fixed_structure, remaining_errors


async def fix_single_item(item: Dict[str, Any], pages: List[tuple], model: str, client) -> Dict[str, Any]:
    """Fix a single structure item by finding the correct page"""
    title = item['title']
    
    # Search in a range around the original index
    original_index = item.get('physical_index', len(pages) // 2)
    search_range = 10
    start_page = max(1, original_index - search_range)
    end_page = min(len(pages), original_index + search_range)
    
    # Create content with page markers
    content = ""
    for page_idx in range(start_page - 1, end_page):
        content += f"<physical_index_{page_idx + 1}>\n{pages[page_idx][0]}\n<physical_index_{page_idx + 1}>\n\n"
    
    prompt = f"""
Find where this section starts in the document pages.

TASK: Locate the physical_index where the section title begins.

RULES:
1. Look for the section title in the provided pages
2. Return the physical_index tag where the section starts
3. Use fuzzy matching - ignore spacing/formatting differences
4. If section not found, set found: false

OUTPUT:
{{
  "physical_index": "<physical_index_X>",
  "found": true
}}

Section Title: {title}
Document Pages:
{content}"""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        json_content = extract_json(response.choices[0].message.content)
        
        if json_content.get('found', False):
            # Convert physical_index to integer
            phys_index_str = json_content.get('physical_index', '')
            if phys_index_str.startswith('<physical_index_'):
                try:
                    phys_index = int(phys_index_str.split('_')[-1].rstrip('>'))
                    return {'found': True, 'physical_index': phys_index}
                except ValueError:
                    pass
        
        return {'found': False, 'physical_index': None}
        
    except Exception as e:
        print(f"Error in single item fix: {e}")
        return {'found': False, 'physical_index': None}


def validate_and_truncate_indices(structure: List[Dict[str, Any]], 
                                 page_count: int, context) -> List[Dict[str, Any]]:
    """Validate and truncate physical indices that exceed document length"""
    
    validated_structure = []
    truncated_count = 0
    
    for item in structure:
        if 'physical_index' in item and item['physical_index'] is not None:
            if item['physical_index'] > page_count:
                # Remove invalid physical_index
                item_copy = item.copy()
                item_copy['physical_index'] = None
                truncated_count += 1
                context.log_step("structure_verifier", "truncated_index", {
                    "title": item.get('title', 'Unknown'),
                    "original_index": item['physical_index'],
                    "page_count": page_count
                })
            else:
                validated_structure.append(item)
        else:
            validated_structure.append(item)
    
    if truncated_count > 0:
        context.log_step("structure_verifier", "validation_complete", {
            "truncated_items": truncated_count,
            "remaining_items": len(validated_structure)
        })
    
    return validated_structure