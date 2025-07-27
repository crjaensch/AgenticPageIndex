import json
import openai
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from core.context import PageIndexContext
from core.exceptions import PageIndexToolError
from core.utils import create_recovery_suggestions
from core.async_utils import run_async_safe
from core.llm_batch_utils import batch_summarize_nodes

def safe_int_conversion(value) -> int:
    """Safely convert physical_index to integer, handling string formats"""
    if value is None:
        return None
    
    if isinstance(value, int):
        return value
    
    if isinstance(value, str):
        if value.startswith('<physical_index_'):
            try:
                return int(value.split('_')[-1].rstrip('>').strip())
            except ValueError:
                return None
        else:
            try:
                return int(value)
            except ValueError:
                return None
    
    return None

def structure_processor_tool(context: Dict[str, Any], 
                           enhancements: List[str] = None) -> Dict[str, Any]:
    """
    Generate final tree structure with optional enhancements
    
    Args:
        context: Serialized PageIndexContext with structure_verified
        enhancements: List of enhancements to apply (node_ids, summaries, etc.)
        
    Returns:
        Updated context with structure_final populated
    """
    
    try:
        # Reconstruct context using consistent approach
        context = PageIndexContext.from_dict(context)
        
        # Setup logging directory
        log_dir = Path(context.config.global_config.log_dir) / context.session_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log step start
        context.log_step("structure_processor", "started", {"enhancements": enhancements})
        
        # Load data
        context.log_step("structure_processor", "loading_data")
        pages = context.load_pages()
        structure_verified = context.structure_verified
        
        if not structure_verified:
            raise PageIndexToolError("No verified structure available for processing")

        # Get configuration
        context.log_step("structure_processor", "loading_config")
        processor_config = context.config.structure_processor
        model = context.config.global_config.model
        
        # Apply default enhancements if none specified
        context.log_step("structure_processor", "determining_enhancements")
        if enhancements is None:
            enhancements = []
            if processor_config.if_add_node_id == "yes":
                enhancements.append("node_ids")
            if processor_config.if_add_node_summary == "yes":
                enhancements.append("summaries")
            if processor_config.if_add_node_text == "yes":
                enhancements.append("full_text")
            if processor_config.if_add_doc_description == "yes":
                enhancements.append("context")
        
        # Add preface if needed
        context.log_step("structure_processor", "adding_preface")
        structure_with_preface = add_preface_if_needed(structure_verified, context)
        
        # Check for start appearance (beginning of sections)
        context.log_step("structure_processor", "checking_section_starts")
        structure_with_starts = run_async_safe(
            check_section_starts(structure_with_preface, pages, model, context)
        )
        
        # Filter valid items and convert to tree
        context.log_step("structure_processor", "building_tree")
        tree_structure = build_tree_structure(structure_with_starts, len(pages), context)
        
        # Process large nodes recursively if needed
        context.log_step("structure_processor", "processing_large_nodes")
        final_tree = run_async_safe(
            process_large_nodes_recursive(tree_structure, pages, processor_config, model, context)
        )
        
        # Apply enhancements
        context.log_step("apply_enhancements", "starting_enhancements")
        enhanced_structure = apply_enhancements(final_tree, pages, enhancements, model, context)
        
        # Build final result
        context.log_step("structure_processor", "building_final_result")
        result = {
            "doc_name": context.pdf_metadata.get("pdf_name", "Unknown"),
            "structure": enhanced_structure
        }
        
        # Add document description if requested
        if "doc_description" in enhancements:
            context.log_step("structure_processor", "generating_doc_description")
            result["doc_description"] = run_async_safe(generate_document_description(enhanced_structure, model))
        
        context.structure_final = result
        
        # Log success
        context.log_step("structure_processor", "completed", {
            "final_nodes": count_nodes(enhanced_structure),
            "enhancements_applied": enhancements
        })
        
        # Save checkpoint
        context.save_checkpoint(log_dir)
        
        return {
            "success": True,
            "context": context.to_dict(),
            "confidence": 1.0,
            "metrics": {
                "final_nodes": count_nodes(enhanced_structure),
                "enhancements_applied": len(enhancements),
                "tree_depth": calculate_tree_depth(enhanced_structure)
            },
            "errors": [],
            "suggestions": []
        }
        
    except Exception as e:
        # Handle errors
        if 'context' in locals():
            context.log_step("structure_processor", "failed", {"error": str(e)})
            context.save_checkpoint(log_dir)
        
        suggestions = create_recovery_suggestions("structure_processing", str(e))
        suggestions.extend([
            "Try processing without enhancements",
            "Check if verified structure is valid",
            "Consider reducing max_page_num_each_node for large documents"
        ])
        
        return {
            "success": False,
            "context": context.to_dict() if 'context' in locals() else context,
            "confidence": 0.0,
            "metrics": {},
            "errors": [str(e)],
            "suggestions": suggestions
        }


def add_preface_if_needed(structure: List[Dict[str, Any]], context) -> List[Dict[str, Any]]:
    """Add a preface node if the document doesn't start on page 1"""
    context.log_step("add_preface_if_needed", "checking_preface")
    # This is a potential crash point if physical_index is None.
    # Using .get() with a default prevents a crash but might hide the logic error.
    # The safest check is to see if the key exists and is not None.
    first_item_index = safe_int_conversion(structure[0].get('physical_index')) if structure else None
    
    if not structure or first_item_index is None or first_item_index > 1:
        context.log_step("add_preface_if_needed", "adding_preface_node")
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        
        return [preface_node] + structure
    
    return structure


async def check_section_starts(items: List[Dict[str, Any]], pages: List[tuple], model: str, context) -> List[Dict[str, Any]]:
    """Check if section titles appear at the start of their pages"""
    
    async def check_appearance(item, pages, model):
        """Check if item title appears at the start of its page"""
        page_idx = safe_int_conversion(item.get('physical_index'))
        
        # Skip items without valid physical_index
        # This check prevents crashes when physical_index is None or invalid
        if not page_idx or not (0 < page_idx <= len(pages)):
            item['appear_start'] = 'unknown'
            return item

        page_content = pages[page_idx - 1][0]
        title = item.get('title', '')
        
        # Simple check: title is in the first 150 chars of the page
        if title.lower() in page_content[:150].lower():
            item['appear_start'] = 'yes'
        else:
            item['appear_start'] = 'no'
            
        return item

    tasks = [check_appearance(item, pages, model) for item in items]
    return await asyncio.gather(*tasks)


def build_tree_structure(items: List[Dict[str, Any]], total_pages: int, context) -> List[Dict[str, Any]]:
    """Build hierarchical tree structure from flat list"""
    
    context.log_step("build_tree_structure", "starting_build", {"item_count": len(items)})
    # Add start and end indices without discarding items
    for i, item in enumerate(items):
        context.log_step("build_tree_structure", "processing_item", {"item_title": item.get("title"), "item_structure": item.get("structure")})
        item['start_index'] = safe_int_conversion(item.get('physical_index'))
        if i < len(items) - 1:
            next_physical_index = safe_int_conversion(items[i + 1].get('physical_index'))
            if next_physical_index is not None:
                if items[i + 1].get('appear_start') == 'yes':
                    item['end_index'] = next_physical_index - 1
                else:
                    item['end_index'] = next_physical_index
            else:
                # If the next item has no page, we cannot determine the end index from it.
                # We can either leave it as null or set it to the same as the start index.
                item['end_index'] = item['start_index']
        else:
            item['end_index'] = total_pages
    
    # DO NOT filter out items. The tree should be built from all items,
    # even those without a determined page number.
    # valid_items = [item for item in items if item.get('start_index') is not None]
    
    # Convert to tree structure
    tree = list_to_tree(items)
    return tree if tree else items


def list_to_tree(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert flat list to hierarchical tree structure"""
    
    def get_parent_structure(structure):
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # Create nodes and track relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        # Copy other attributes
        for key, value in item.items():
            if key not in ['title', 'start_index', 'end_index']:
                node[key] = value
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure and parent_structure in nodes:
            nodes[parent_structure]['nodes'].append(node)
        else:
            root_nodes.append(node)
    
    # Clean empty nodes arrays
    def clean_node(node):
        if not node.get('nodes'):
            node.pop('nodes', None)
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    return [clean_node(node) for node in root_nodes]


async def process_large_nodes_recursive(tree: List[Dict[str, Any]], pages: List[tuple],
                                       config, model: str, context) -> List[Dict[str, Any]]:
    """Recursively process large nodes by subdividing them"""
    
    async def process_node(node):
        start_idx = node.get('start_index')
        end_idx = node.get('end_index')
        
        # Log the node being processed
        context.log_step("process_large_nodes", "processing_node", {"node_title": node.get("title"), "start": start_idx, "end": end_idx})

        # Only process if indices are valid and within the bounds of the document
        if start_idx is not None and end_idx is not None and 0 < start_idx <= end_idx <= len(pages):
            node_pages = pages[start_idx-1:end_idx]
            token_count = sum(page[1] for page in node_pages)
            page_count = end_idx - start_idx + 1
            
            max_pages = config.max_page_num_each_node
            max_tokens = config.max_token_num_each_node
            
            if page_count > max_pages and token_count >= max_tokens:
                context.log_step("structure_processor", "subdividing_large_node", {
                    "title": node.get('title', 'Unknown'),
                    "pages": page_count,
                    "tokens": token_count
                })
                
                # Generate sub-structure for this node
                # This would use the structure extraction logic for the node's content
                # For now, we'll keep the node as is to avoid infinite recursion
                pass
        
        # Process child nodes if they exist
        if 'nodes' in node and node['nodes']:
            tasks = [process_node(child) for child in node['nodes']]
            await asyncio.gather(*tasks)
        
        return node
    
    # Process all top-level nodes
    tasks = [process_node(node) for node in tree]
    processed_tree = await asyncio.gather(*tasks)
    
    return processed_tree


def apply_enhancements(structure: List[Dict[str, Any]], pages: List[tuple], 
                       enhancements: List[str], model: str, context) -> List[Dict[str, Any]]:
    """Apply enhancements to the final tree structure"""
    
    context.log_step("apply_enhancements", "starting_enhancements")
    # Make a deep copy to avoid modifying the original structure
    copied_structure = json.loads(json.dumps(structure))
    
    if "node_ids" in enhancements:
        context.log_step("structure_processor", "adding_node_ids")
        add_node_ids(copied_structure)
    
    if "node_text" in enhancements:
        context.log_step("structure_processor", "adding_node_text")
        add_node_text_recursive(copied_structure, pages, context)
    
    if "summaries" in enhancements:
        context.log_step("structure_processor", "generating_summaries")
        if "node_text" not in enhancements:
            add_node_text_recursive(copied_structure, pages, context)
        
        # Generate summaries asynchronously
        run_async_safe(add_summaries(copied_structure, pages, model, context))
        
        if "node_text" not in enhancements:
            remove_node_text(copied_structure)
    
    return copied_structure


def add_node_ids(structure: List[Dict[str, Any]], node_id: int = 0) -> int:
    """Add unique node IDs to structure"""
    
    def add_id_recursive(nodes, current_id):
        for node in nodes:
            node['node_id'] = str(current_id).zfill(4)
            current_id += 1
            if 'nodes' in node:
                current_id = add_id_recursive(node['nodes'], current_id)
        return current_id
    
    return add_id_recursive(structure, node_id)


def add_node_text_recursive(structure: List[Dict[str, Any]], pages: List[tuple], context):
    """Recursively add text content to each node using page-level extraction"""
    
    context.log_step("add_node_text_recursive", "starting_text_addition")
    
    def add_text(node, pages):
        """Add text content to a node using complete pages"""
        start = node.get("start_index")
        end = node.get("end_index")
        title = node.get("title", "")
        
        # Log the node being processed
        context.log_step("add_node_text_recursive", "adding_text", {
            "node_title": title, "start": start, "end": end
        })

        # Only add text if indices are valid and within the bounds of the document
        if start is not None and end is not None and 0 < start <= end <= len(pages):
            # Extract complete text from all pages in the range
            node["text"] = "".join(p[0] for p in pages[start-1:end])
            
            context.log_step("add_node_text_recursive", "text_added", {
                "node_title": title,
                "pages_included": f"{start}-{end}",
                "text_length": len(node["text"])
            })
            
        if 'nodes' in node:
            for child in node['nodes']:
                add_text(child, pages)
    
    for node in structure:
        add_text(node, pages)


def remove_node_text(structure: List[Dict[str, Any]]):
    """Remove text content from nodes"""
    
    def remove_text_recursive(nodes):
        for node in nodes:
            node.pop('text', None)
            if 'nodes' in node:
                remove_text_recursive(node['nodes'])
    
    remove_text_recursive(structure)


async def add_summaries(structure: List[Dict[str, Any]], pages: List[tuple], model: str, context):
    """Generate summaries for all nodes with text using efficient batching"""
    
    context.log_step("add_summaries", "starting_summaries")
    
    def collect_nodes(nodes):
        all_nodes = []
        for node in nodes:
            all_nodes.append(node)
            if 'nodes' in node:
                all_nodes.extend(collect_nodes(node['nodes']))
        return all_nodes
    
    # Collect all nodes that need summaries
    all_nodes = collect_nodes(structure)
    nodes_to_summarize = []
    
    # Prepare nodes for summarization with robust identification
    for i, node in enumerate(all_nodes):
        # Check if node has text content or can get text content
        text = None
        if 'text' in node:
            text = node['text']
        else:
            # Try to get text content for the node
            text = get_text_for_node(node, pages, context)
        
        if text and len(text.split()) > 20:  # Only substantial text
            # Create a robust node entry for batching
            node_entry = {
                'node_index': i,  # Use index as reliable identifier
                'node_ref': node,  # Keep reference to original node
                'text': text,
                'title': node.get('title', f'Section_{i}'),  # Fallback title
                'original_title': node.get('title', '')  # Keep original for logging
            }
            nodes_to_summarize.append(node_entry)
            context.log_step("add_summaries", "preparing_node_for_batch", {
                "node_index": i, 
                "node_title": node.get("title", f'Section_{i}')
            })
    
    if nodes_to_summarize:
        context.log_step("add_summaries", "batching_summaries", {"node_count": len(nodes_to_summarize)})
        
        try:
            # Use batch processing for all summaries
            summaries = await batch_summarize_nodes(nodes_to_summarize, model)
            
            # Apply summaries back to original nodes using reliable mapping
            success_count = 0
            for node_entry in nodes_to_summarize:
                node_index = node_entry['node_index']
                title = node_entry['title']
                original_node = node_entry['node_ref']
                
                if title in summaries and summaries[title].strip():
                    original_node['summary'] = summaries[title]
                    success_count += 1
                    context.log_step("add_summaries", "summary_applied", {
                        "node_index": node_index,
                        "node_title": node_entry['original_title']
                    })
                else:
                    # Fallback: try individual processing for this node
                    context.log_step("add_summaries", "batch_failed_fallback", {
                        "node_index": node_index,
                        "node_title": node_entry['original_title']
                    })
                    
                    # Individual fallback processing
                    try:
                        client = openai.AsyncOpenAI()
                        prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

                        Partial Document Text: {node_entry['text']}
                        
                        Directly return the description, do not include any other text.
                        """
                        
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0,
                        )
                        
                        original_node['summary'] = response.choices[0].message.content
                        success_count += 1
                        context.log_step("add_summaries", "individual_fallback_success", {
                            "node_index": node_index
                        })
                    except Exception as fallback_error:
                        original_node['summary'] = "Summary generation failed"
                        context.log_step("add_summaries", "individual_fallback_failed", {
                            "node_index": node_index,
                            "error": str(fallback_error)
                        })
            
            context.log_step("add_summaries", "batch_completed", {
                "total_nodes": len(nodes_to_summarize),
                "successful_summaries": success_count
            })
        
        except Exception as e:
            context.log_step("add_summaries", "batch_error", {"error": str(e)})
            # Complete fallback: process all nodes individually
            context.log_step("add_summaries", "falling_back_to_individual_processing")
            
            for node_entry in nodes_to_summarize:
                try:
                    client = openai.AsyncOpenAI()
                    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

                    Partial Document Text: {node_entry['text']}
                    
                    Directly return the description, do not include any other text.
                    """
                    
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                    )
                    
                    node_entry['node_ref']['summary'] = response.choices[0].message.content
                except Exception as individual_error:
                    node_entry['node_ref']['summary'] = "Summary generation failed"
                    context.log_step("add_summaries", "individual_processing_failed", {
                        "node_index": node_entry['node_index'],
                        "error": str(individual_error)
                    })
    
    context.log_step("add_summaries", "completed_summaries")


def get_text_for_node(node: Dict[str, Any], pages: List[tuple], context) -> str:
    """Get text content for a specific node"""
    start = node.get("start_index")
    end = node.get("end_index")
    
    context.log_step("get_text_for_node", "getting_text", {"node_title": node.get("title"), "start": start, "end": end})
    # Only return text if indices are valid and within the bounds of the document
    if start is not None and end is not None and 0 < start <= end <= len(pages):
        return "".join(p[0] for p in pages[start-1:end])
        
    return ""


async def generate_document_description(structure: List[Dict[str, Any]], model: str) -> str:
    """Generate overall document description"""
    client = openai.OpenAI()
    
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {json.dumps(structure, indent=2)}
    
    Directly return the description, do not include any other text.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating document description: {e}")
        return "Document description generation failed"


def count_nodes(structure: List[Dict[str, Any]]) -> int:
    """Count total number of nodes in structure"""
    count = 0
    
    def count_recursive(nodes):
        nonlocal count
        for node in nodes:
            count += 1
            if 'nodes' in node:
                count_recursive(node['nodes'])
    
    count_recursive(structure)
    return count


def calculate_tree_depth(structure: List[Dict[str, Any]]) -> int:
    """Calculate maximum depth of tree structure"""
    
    def depth_recursive(nodes, current_depth=0):
        if not nodes:
            return current_depth
        
        max_depth = current_depth
        for node in nodes:
            if 'nodes' in node:
                depth = depth_recursive(node['nodes'], current_depth + 1)
                max_depth = max(max_depth, depth)
            else:
                max_depth = max(max_depth, current_depth + 1)
        
        return max_depth
    
    return depth_recursive(structure)