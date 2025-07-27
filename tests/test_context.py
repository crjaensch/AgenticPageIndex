"""
Unit tests for the context module
"""

import unittest
import tempfile
import json
from pathlib import Path

from core.context import PageIndexContext
from core.config_schema import PageIndexConfig


class TestPageIndexContext(unittest.TestCase):
    """Unit tests for PageIndexContext class"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary directory for test logs
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="pageindex_test_"))
        
        # Create a real config instance
        self.config = PageIndexConfig()
        
        # Create test pages data
        self.test_pages = [
            ("Page 1 content here", 100),
            ("Page 2 content here", 150),
            ("Page 3 content here", 120)
        ]
        
        # Create test structure data
        self.test_structure = [
            {"title": "Introduction", "structure": "1", "physical_index": 1},
            {"title": "Methods", "structure": "2", "physical_index": 2},
            {"title": "Results", "structure": "3", "physical_index": 3}
        ]
        
        # Create test TOC info
        self.test_toc_info = {
            "found": True,
            "has_page_numbers": True,
            "content": "1. Introduction...1\n2. Methods...2\n3. Results...3"
        }
    
    def test_context_initialization(self):
        """Test PageIndexContext initialization"""
        context = PageIndexContext(self.config)
        
        # Check that all attributes are initialized correctly
        self.assertIsInstance(context.session_id, str)
        self.assertEqual(context.config, self.config)
        self.assertEqual(context.pdf_metadata, {})
        self.assertIsNone(context.pages_file)
        self.assertEqual(context.toc_info, {})
        self.assertEqual(context.structure_raw, [])
        self.assertEqual(context.structure_verified, [])
        self.assertEqual(context.structure_final, {})
        self.assertEqual(context.processing_log, [])
        self.assertEqual(context.current_step, "initialized")
    
    def test_log_step(self):
        """Test log_step method"""
        context = PageIndexContext(self.config)
        
        # Log a step
        context.log_step("pdf_parser", "started", {"pages": 10})
        
        # Check that the step was logged correctly
        self.assertEqual(len(context.processing_log), 1)
        log_entry = context.processing_log[0]
        self.assertEqual(log_entry["tool"], "pdf_parser")
        self.assertEqual(log_entry["status"], "started")
        self.assertEqual(log_entry["details"], {"pages": 10})
        self.assertIn("timestamp", log_entry)
        self.assertEqual(context.current_step, "pdf_parser_started")
    
    def test_save_and_load_pages(self):
        """Test save_pages and load_pages methods"""
        context = PageIndexContext(self.config)
        
        # Save pages
        context.save_pages(self.test_pages, self.test_log_dir)
        
        # Check that pages_file was set correctly
        self.assertIsNotNone(context.pages_file)
        pages_path = Path(context.pages_file)
        self.assertTrue(pages_path.exists())
        
        # Load pages
        loaded_pages = context.load_pages()
        # When JSON loads data, tuples become lists, so we need to convert them back
        converted_pages = [tuple(page) for page in loaded_pages]
        self.assertEqual(converted_pages, self.test_pages)
    
    def test_load_pages_empty(self):
        """Test load_pages when no pages file exists"""
        context = PageIndexContext(self.config)
        
        # Load pages when pages_file is None
        loaded_pages = context.load_pages()
        self.assertEqual(loaded_pages, [])
    
    def test_save_checkpoint(self):
        """Test save_checkpoint method"""
        context = PageIndexContext(self.config)
        
        # Add some data to the context
        context.pdf_metadata = {"pdf_name": "test.pdf", "total_pages": 10}
        context.toc_info = self.test_toc_info
        context.structure_raw = self.test_structure
        context.log_step("pdf_parser", "completed")
        
        # Save checkpoint
        context.save_checkpoint(self.test_log_dir)
        
        # Check that checkpoint file was created
        checkpoint_path = self.test_log_dir / f"{context.session_id}_checkpoint.json"
        self.assertTrue(checkpoint_path.exists())
        
        # Load and verify checkpoint
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        self.assertEqual(checkpoint_data["session_id"], context.session_id)
        self.assertEqual(checkpoint_data["pdf_metadata"], context.pdf_metadata)
        self.assertEqual(checkpoint_data["toc_info"], context.toc_info)
        self.assertEqual(checkpoint_data["structure_raw"], context.structure_raw)
        self.assertEqual(len(checkpoint_data["processing_log"]), 1)  # Only last 5 steps
    
    def test_save_checkpoint_with_pages(self):
        """Test save_checkpoint method with pages data"""
        context = PageIndexContext(self.config)
        
        # Add pages data
        context.save_pages(self.test_pages, self.test_log_dir)
        
        # Save checkpoint with pages
        context.save_checkpoint(self.test_log_dir, include_pages=True)
        
        # Check that checkpoint file was created
        checkpoint_path = self.test_log_dir / f"{context.session_id}_checkpoint.json"
        self.assertTrue(checkpoint_path.exists())
        
        # Load and verify checkpoint
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
        
        self.assertIn("pages_data", checkpoint_data)
        # When JSON loads data, tuples become lists, so we need to convert them back
        converted_pages = [tuple(page) for page in checkpoint_data["pages_data"]]
        self.assertEqual(converted_pages, self.test_pages)
    
    def test_to_dict(self):
        """Test to_dict method"""
        context = PageIndexContext(self.config)
        
        # Add some data to the context
        context.pdf_metadata = {"pdf_name": "test.pdf", "total_pages": 10}
        context.toc_info = self.test_toc_info
        context.structure_raw = self.test_structure
        context.structure_verified = self.test_structure
        context.structure_final = {"title": "Document", "children": self.test_structure}
        
        # Add multiple log entries
        for i in range(10):
            context.log_step("tool", f"step_{i}")
        
        # Convert to dictionary
        context_dict = context.to_dict()
        
        # Check that all expected fields are present
        expected_fields = [
            "session_id", "config", "pdf_metadata", "pages_file", "toc_info",
            "structure_raw", "structure_verified", "structure_final", 
            "processing_log", "current_step"
        ]
        
        for field in expected_fields:
            self.assertIn(field, context_dict)
        
        # Check that only last 5 log entries are included
        self.assertEqual(len(context_dict["processing_log"]), 5)
        
        # Check that config is serialized correctly
        self.assertIsInstance(context_dict["config"], dict)
    
    def test_from_dict(self):
        """Test from_dict method"""
        # Create original context
        original_context = PageIndexContext(self.config)
        original_context.pdf_metadata = {"pdf_name": "test.pdf", "total_pages": 10}
        original_context.toc_info = self.test_toc_info
        original_context.structure_raw = self.test_structure
        original_context.log_step("pdf_parser", "completed")
        
        # Convert to dictionary
        context_dict = original_context.to_dict()
        
        # Create new context from dictionary
        new_context = PageIndexContext.from_dict(context_dict)
        
        # Check that all data was restored correctly
        self.assertEqual(new_context.session_id, original_context.session_id)
        self.assertEqual(new_context.pdf_metadata, original_context.pdf_metadata)
        self.assertEqual(new_context.toc_info, original_context.toc_info)
        self.assertEqual(new_context.structure_raw, original_context.structure_raw)
        self.assertEqual(new_context.processing_log, original_context.processing_log)
        self.assertEqual(new_context.current_step, original_context.current_step)
    
    def test_from_dict_with_missing_fields(self):
        """Test from_dict method with missing fields"""
        # Create dictionary with missing fields
        context_dict = {
            "session_id": "test_session",
            "config": {}
        }
        
        # Create context from dictionary
        context = PageIndexContext.from_dict(context_dict)
        
        # Check that missing fields have default values
        self.assertEqual(context.session_id, "test_session")
        self.assertEqual(context.pdf_metadata, {})
        self.assertIsNone(context.pages_file)
        self.assertEqual(context.toc_info, {})
        self.assertEqual(context.structure_raw, [])
        self.assertEqual(context.structure_verified, [])
        self.assertEqual(context.structure_final, {})
        self.assertEqual(context.processing_log, [])
        self.assertEqual(context.current_step, "initialized")


if __name__ == '__main__':
    unittest.main()
