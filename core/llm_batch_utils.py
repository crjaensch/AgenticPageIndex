"""
LLM Batching utilities for efficient token usage and reduced API calls
"""

import asyncio
import openai
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from core.utils import count_tokens


@dataclass
class BatchItem:
    """Single item in a batch request"""
    id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchResult:
    """Result from a batch request"""
    id: str
    result: str
    error: Optional[str] = None


class LLMBatcher:
    """Efficient LLM batching for similar operations with token awareness"""
    
    def __init__(self, model: str = "gpt-4", max_tokens: int = 120000):
        self.model = model
        self.client = openai.AsyncOpenAI()
        self.max_tokens = max_tokens  # Conservative limit for context window
    
    async def batch_summarize(self, items: List[BatchItem]) -> List[BatchResult]:
        """
        Batch multiple text summarization requests into a single LLM call
        
        Args:
            items: List of BatchItem objects with text to summarize
            
        Returns:
            List of BatchResult objects with summaries
        """
        if not items:
            return []
        
        # Build batch prompt with all items
        batch_prompt = self._build_summary_batch_prompt(items)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0,
            )
            
            # Parse batch response
            return self._parse_summary_batch_response(response.choices[0].message.content, items)
            
        except Exception as e:
            # Return error results for all items
            return [BatchResult(id=item.id, result="", error=str(e)) for item in items]
    
    def _build_summary_batch_prompt(self, items: List[BatchItem]) -> str:
        """Build a single prompt for multiple summarization tasks"""
        
        prompt = """You are given multiple document sections to summarize. For each section, generate a description of the main points covered.

Return your response in the following JSON format:
{
  "summaries": [
    {"id": "item_1", "summary": "description of main points"},
    {"id": "item_2", "summary": "description of main points"},
    ...
  ]
}

Document sections to summarize:

"""
        
        for item in items:
            prompt += f"ID: {item.id}\n"
            prompt += f"Text: {item.content}\n\n"
        
        prompt += "Return only the JSON response with summaries for all sections."
        
        return prompt
    
    def _parse_summary_batch_response(self, response: str, items: List[BatchItem]) -> List[BatchResult]:
        """Parse batch response and map back to individual results"""
        try:
            import json
            from core.utils import extract_json
            
            # Try to extract JSON from response using the utility function
            try:
                response_data = extract_json(response)
            except:
                # Fallback to direct JSON parsing
                response_data = json.loads(response)
            
            summaries = response_data.get("summaries", [])
            
            # Validate summaries structure
            if not isinstance(summaries, list):
                raise ValueError("Summaries is not a list")
            
            # Create mapping from response with validation
            summary_map = {}
            for s in summaries:
                if isinstance(s, dict) and "id" in s and "summary" in s:
                    summary_map[s["id"]] = s["summary"]
            
            # Build results in original order
            results = []
            for item in items:
                if item.id in summary_map and summary_map[item.id].strip():
                    # Valid summary found
                    results.append(BatchResult(id=item.id, result=summary_map[item.id]))
                else:
                    # No valid summary found - return error to trigger fallback
                    results.append(BatchResult(id=item.id, result="", error="Summary not found or empty in batch response"))
            
            return results
            
        except Exception as e:
            # Fallback: return error for all items to trigger individual processing
            return [BatchResult(id=item.id, result="", error=f"Parse error: {str(e)}") for item in items]
    
    async def batch_extract_structure(self, items: List[BatchItem], operation_type: str) -> List[BatchResult]:
        """
        Batch multiple structure extraction requests
        
        Args:
            items: List of BatchItem objects with content to process
            operation_type: Type of extraction ("toc_transform", "physical_indices", etc.)
            
        Returns:
            List of BatchResult objects with extracted structures
        """
        if not items:
            return []
        
        # Build operation-specific batch prompt
        batch_prompt = self._build_extraction_batch_prompt(items, operation_type)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": batch_prompt}],
                temperature=0,
            )
            
            return self._parse_extraction_batch_response(response.choices[0].message.content, items)
            
        except Exception as e:
            return [BatchResult(id=item.id, result="", error=str(e)) for item in items]
    
    def _build_extraction_batch_prompt(self, items: List[BatchItem], operation_type: str) -> str:
        """Build batch prompt for structure extraction operations"""
        
        base_prompts = {
            "toc_transform": "Transform the following raw TOC content into structured JSON format",
            "physical_indices": "Extract physical page indices for the following TOC items",
            "content_matching": "Match TOC items to content sections"
        }
        
        prompt = f"{base_prompts.get(operation_type, 'Process the following items')}.\n\n"
        prompt += "Return your response in JSON format with an array of results:\n"
        prompt += '{"results": [{"id": "item_1", "result": {...}}, {"id": "item_2", "result": {...}}]}\n\n'
        
        for item in items:
            prompt += f"ID: {item.id}\n"
            prompt += f"Content: {item.content}\n\n"
        
        return prompt
    
    def _parse_extraction_batch_response(self, response: str, items: List[BatchItem]) -> List[BatchResult]:
        """Parse batch extraction response"""
        try:
            import json
            
            response_data = json.loads(response)
            results_data = response_data.get("results", [])
            
            # Create mapping from response
            result_map = {r["id"]: r["result"] for r in results_data}
            
            # Build results in original order
            results = []
            for item in items:
                result = result_map.get(item.id, {})
                results.append(BatchResult(id=item.id, result=json.dumps(result)))
            
            return results
            
        except Exception as e:
            return [BatchResult(id=item.id, result="", error=f"Parse error: {str(e)}") for item in items]
    
    def _split_items_by_token_limit(self, items: List[BatchItem], base_prompt: str) -> List[List[BatchItem]]:
        """Split items into batches that respect token limits"""
        if not items:
            return []
        
        batches = []
        current_batch = []
        current_tokens = count_tokens(base_prompt, self.model)
        
        for item in items:
            item_tokens = count_tokens(f"ID: {item.id}\nContent: {item.content}\n\n", self.model)
            
            # If adding this item would exceed limit, start new batch
            if current_tokens + item_tokens > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = [item]
                current_tokens = count_tokens(base_prompt, self.model) + item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens
        
        # Add final batch if not empty
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def batch_toc_operations(self, items: List[BatchItem], operation_type: str) -> List[BatchResult]:
        """Token-aware batching for TOC operations with automatic splitting"""
        if not items:
            return []
        
        # Define operation-specific prompts
        operation_prompts = {
            "transform_toc": "Transform the following TOC content items into structured JSON format.",
            "extract_indices": "Extract physical indices for the following TOC items from document content.",
            "match_content": "Match the following TOC items to content sections and add physical indices."
        }
        
        base_prompt = operation_prompts.get(operation_type, "Process the following items.")
        base_prompt += "\n\nReturn response in JSON format: {\"results\": [{\"id\": \"item_1\", \"result\": {...}}]}\n\n"
        
        # Split items into token-aware batches
        batches = self._split_items_by_token_limit(items, base_prompt)
        
        all_results = []
        for batch in batches:
            try:
                # Build batch prompt
                batch_prompt = base_prompt
                for item in batch:
                    batch_prompt += f"ID: {item.id}\n"
                    batch_prompt += f"Content: {item.content}\n\n"
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": batch_prompt}],
                    temperature=0,
                )
                
                batch_results = self._parse_extraction_batch_response(
                    response.choices[0].message.content, batch
                )
                all_results.extend(batch_results)
                
            except Exception as e:
                # Add error results for this batch
                error_results = [BatchResult(id=item.id, result="", error=str(e)) for item in batch]
                all_results.extend(error_results)
        
        return all_results


# Convenience functions for easy integration
async def batch_summarize_nodes(nodes_with_text: List[Dict[str, Any]], model: str) -> Dict[str, str]:
    """
    Convenience function to batch summarize multiple nodes
    
    Args:
        nodes_with_text: List of nodes with 'text' and 'title' fields
        model: LLM model to use
        
    Returns:
        Dictionary mapping node titles to summaries
    """
    if not nodes_with_text:
        return {}
    
    batcher = LLMBatcher(model)
    
    # Prepare batch items with robust identification
    batch_items = []
    for i, node in enumerate(nodes_with_text):
        text = node.get('text', '')
        # Use the title from the node entry (which has fallback logic)
        title = node.get('title', f'node_{i}')
        
        if text and len(text.split()) > 20:  # Only substantial text
            batch_items.append(BatchItem(
                id=title,
                content=text,
                metadata={'node_index': i, 'original_title': node.get('original_title', '')}
            ))
    
    if not batch_items:
        return {}
    
    # Execute batch with better error handling
    try:
        results = await batcher.batch_summarize(batch_items)
        
        # Build result dictionary, only including successful results
        summary_dict = {}
        for result in results:
            if not result.error and result.result.strip():
                summary_dict[result.id] = result.result
        
        return summary_dict
        
    except Exception as e:
        print(f"Error in batch_summarize_nodes: {e}")
        return {}  # Return empty dict to trigger individual fallback
