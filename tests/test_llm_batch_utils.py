"""
Unit tests for the llm_batch_utils module
"""

import unittest
import asyncio
from unittest.mock import AsyncMock, patch

from core.llm_batch_utils import LLMBatcher, BatchItem, BatchResult, batch_summarize_nodes


class TestLLMBatcher(unittest.TestCase):
    """Unit tests for LLMBatcher class"""
    
    def setUp(self):
        """Set up test data"""
        self.batcher = LLMBatcher(model="gpt-4")
        
        # Create test batch items
        self.test_items = [
            BatchItem(id="item_1", content="This is the first test content for summarization."),
            BatchItem(id="item_2", content="This is the second test content for summarization."),
            BatchItem(id="item_3", content="This is the third test content for summarization.")
        ]
        
        # Create test extraction items
        self.extraction_items = [
            BatchItem(id="extract_1", content="Raw TOC content 1"),
            BatchItem(id="extract_2", content="Raw TOC content 2")
        ]
    
    def test_batch_item_initialization(self):
        """Test BatchItem initialization"""
        item = BatchItem(id="test_id", content="test content", metadata={"key": "value"})
        
        self.assertEqual(item.id, "test_id")
        self.assertEqual(item.content, "test content")
        self.assertEqual(item.metadata, {"key": "value"})
        
        # Test without metadata
        item_no_meta = BatchItem(id="test_id", content="test content")
        self.assertIsNone(item_no_meta.metadata)
    
    def test_batch_result_initialization(self):
        """Test BatchResult initialization"""
        result = BatchResult(id="test_id", result="test result")
        
        self.assertEqual(result.id, "test_id")
        self.assertEqual(result.result, "test result")
        self.assertIsNone(result.error)
        
        # Test with error
        result_with_error = BatchResult(id="test_id", result="", error="test error")
        self.assertEqual(result_with_error.error, "test error")
    
    def test_llm_batcher_initialization(self):
        """Test LLMBatcher initialization"""
        batcher = LLMBatcher(model="gpt-3.5-turbo", max_tokens=60000)
        
        self.assertEqual(batcher.model, "gpt-3.5-turbo")
        self.assertEqual(batcher.max_tokens, 60000)
        
        # Test default values
        batcher_default = LLMBatcher()
        self.assertEqual(batcher_default.model, "gpt-4")
        self.assertEqual(batcher_default.max_tokens, 120000)
    
    def test_build_summary_batch_prompt(self):
        """Test _build_summary_batch_prompt method"""
        prompt = self.batcher._build_summary_batch_prompt(self.test_items)
        
        # Check that prompt contains expected elements
        self.assertIn("You are given multiple document sections to summarize", prompt)
        self.assertIn("item_1", prompt)
        self.assertIn("This is the first test content for summarization.", prompt)
        self.assertIn("item_2", prompt)
        self.assertIn("This is the second test content for summarization.", prompt)
        self.assertIn("Return only the JSON response with summaries for all sections.", prompt)
    
    def test_build_extraction_batch_prompt(self):
        """Test _build_extraction_batch_prompt method"""
        prompt = self.batcher._build_extraction_batch_prompt(self.extraction_items, "toc_transform")
        
        # Check that prompt contains expected elements
        self.assertIn("Transform the following raw TOC content into structured JSON format", prompt)
        self.assertIn("extract_1", prompt)
        self.assertIn("Raw TOC content 1", prompt)
        self.assertIn("extract_2", prompt)
        self.assertIn("Raw TOC content 2", prompt)
    
    def test_parse_summary_batch_response_valid(self):
        """Test _parse_summary_batch_response with valid response"""
        response = '{"summaries": [{"id": "item_1", "summary": "Summary 1"}, {"id": "item_2", "summary": "Summary 2"}]}'
        
        results = self.batcher._parse_summary_batch_response(response, self.test_items[:2])
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "item_1")
        self.assertEqual(results[0].result, "Summary 1")
        self.assertIsNone(results[0].error)
        self.assertEqual(results[1].id, "item_2")
        self.assertEqual(results[1].result, "Summary 2")
        self.assertIsNone(results[1].error)
    
    def test_parse_summary_batch_response_with_extraction(self):
        """Test _parse_summary_batch_response with extraction-style response"""
        response = '{"results": [{"id": "item_1", "result": {"summary": "Summary 1"}}, {"id": "item_2", "result": {"summary": "Summary 2"}}]}'
        
        results = self.batcher._parse_summary_batch_response(response, self.test_items[:2])
        
        # Should return error results since it doesn't match expected format
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.error is not None or result.result == "")
    
    def test_parse_summary_batch_response_invalid(self):
        """Test _parse_summary_batch_response with invalid response"""
        response = 'Invalid JSON response'
        
        results = self.batcher._parse_summary_batch_response(response, self.test_items[:2])
        
        # Should return error results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.error is not None)
    
    def test_parse_extraction_batch_response_valid(self):
        """Test _parse_extraction_batch_response with valid response"""
        response = '{"results": [{"id": "extract_1", "result": {"key": "value1"}}, {"id": "extract_2", "result": {"key": "value2"}}]}'
        
        results = self.batcher._parse_extraction_batch_response(response, self.extraction_items)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "extract_1")
        self.assertEqual(results[0].result, '{"key": "value1"}')
        self.assertIsNone(results[0].error)
        self.assertEqual(results[1].id, "extract_2")
        self.assertEqual(results[1].result, '{"key": "value2"}')
        self.assertIsNone(results[1].error)
    
    def test_parse_extraction_batch_response_invalid(self):
        """Test _parse_extraction_batch_response with invalid response"""
        response = 'Invalid JSON response'
        
        results = self.batcher._parse_extraction_batch_response(response, self.extraction_items)
        
        # Should return error results
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.error is not None)
    
    def test_split_items_by_token_limit(self):
        """Test _split_items_by_token_limit method"""
        # Create items with known token counts
        items = [
            BatchItem(id="item_1", content="Short content"),  # ~3 tokens
            BatchItem(id="item_2", content="Another short content"),  # ~4 tokens
            BatchItem(id="item_3", content="Yet another short content"),  # ~5 tokens
        ]
        
        # Test with very low token limit to force splitting
        batcher_low_limit = LLMBatcher(max_tokens=10)
        batches = batcher_low_limit._split_items_by_token_limit(items, "Base prompt")
        
        # Should split into multiple batches
        self.assertGreater(len(batches), 1)
        
        # Test with high token limit to keep all in one batch
        batcher_high_limit = LLMBatcher(max_tokens=1000)
        batches = batcher_high_limit._split_items_by_token_limit(items, "Base prompt")
        
        # Should keep all in one batch
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 3)
    
    def test_split_items_by_token_limit_empty(self):
        """Test _split_items_by_token_limit with empty items"""
        batches = self.batcher._split_items_by_token_limit([], "Base prompt")
        self.assertEqual(batches, [])


class TestBatchSummarizeNodes(unittest.TestCase):
    """Unit tests for batch_summarize_nodes function"""
    
    def setUp(self):
        """Set up test data"""
        self.test_nodes = [
            {"title": "Introduction", "text": "This is a long introduction with substantial content that should be summarized. It has more than twenty words to pass the threshold for batch processing."},
            {"title": "Methods", "text": "This section describes the methods used in the research with detailed explanations. It also has more than twenty words to be included in batch processing."},
            {"title": "Results", "text": ""},  # Empty text
            {"title": "Short", "text": "Hi"}  # Too short
        ]
    
    @patch('core.llm_batch_utils.LLMBatcher')
    def test_batch_summarize_nodes_success(self, mock_batcher_class):
        """Test batch_summarize_nodes with successful batch processing"""
        # Mock the batcher and its batch_summarize method
        mock_batcher_instance = AsyncMock()
        mock_batcher_class.return_value = mock_batcher_instance
        
        # Mock results
        mock_results = [
            BatchResult(id="Introduction", result="Summary of introduction"),
            BatchResult(id="Methods", result="Summary of methods")
        ]
        mock_batcher_instance.batch_summarize = AsyncMock(return_value=mock_results)
        
        # Run the function
        result = asyncio.run(batch_summarize_nodes(self.test_nodes, "gpt-4"))
        
        # Check that the batcher was called
        mock_batcher_class.assert_called_once_with("gpt-4")
        
        # Check results
        self.assertIsInstance(result, dict)
        self.assertIn("Introduction", result)
        self.assertEqual(result["Introduction"], "Summary of introduction")
        self.assertIn("Methods", result)
        self.assertEqual(result["Methods"], "Summary of methods")
    
    @patch('core.llm_batch_utils.LLMBatcher')
    def test_batch_summarize_nodes_empty_input(self, mock_batcher_class):
        """Test batch_summarize_nodes with empty input"""
        result = asyncio.run(batch_summarize_nodes([], "gpt-4"))
        self.assertEqual(result, {})
        
        # Should not call the batcher at all
        mock_batcher_class.assert_not_called()
    
    @patch('core.llm_batch_utils.LLMBatcher')
    def test_batch_summarize_nodes_with_errors(self, mock_batcher_class):
        """Test batch_summarize_nodes with batch processing errors"""
        # Mock the batcher to raise an exception
        mock_batcher_instance = AsyncMock()
        mock_batcher_class.return_value = mock_batcher_instance
        mock_batcher_instance.batch_summarize = AsyncMock(side_effect=Exception("Test error"))
        
        # Run the function
        result = asyncio.run(batch_summarize_nodes(self.test_nodes, "gpt-4"))
        
        # Should return empty dict on error
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
