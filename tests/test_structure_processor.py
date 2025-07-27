"""
Unit tests for the structure_processor module
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from tools.structure_processor import (
    safe_int_conversion, add_preface_if_needed, build_tree_structure, 
    list_to_tree, apply_enhancements, add_node_ids, add_node_text_recursive,
    remove_node_text, count_nodes, calculate_tree_depth
)
from core.context import PageIndexContext


class TestStructureProcessor(unittest.TestCase):
    """Unit tests for structure_processor functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary directory for test logs
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="pageindex_test_"))
        
        # Create a mock context
        self.mock_context = MagicMock()
        self.mock_context.config = MagicMock()
        self.mock_context.config.structure_processor = MagicMock()
        self.mock_context.config.global_config = MagicMock()
        self.mock_context.config.global_config.model = "gpt-3.5-turbo"
        self.mock_context.session_id = "test_session"
        self.mock_context.pdf_metadata = {"pdf_name": "test.pdf"}
        
        # Mock pages data
        self.mock_pages = [
            ("Page 1 content here", 100),
            ("Page 2 content here", 150),
            ("Page 3 content here", 120),
            ("Page 4 content here", 180),
            ("Page 5 content here", 200)
        ]
        
        # Mock structure data
        self.mock_structure = [
            {"title": "Introduction", "structure": "1", "physical_index": 1},
            {"title": "Methods", "structure": "2", "physical_index": 3},
            {"title": "Results", "structure": "3", "physical_index": 5}
        ]
        
        # Mock tree structure
        self.mock_tree_structure = [
            {
                "title": "Introduction", 
                "structure": "1", 
                "physical_index": 1,
                "start_index": 1,
                "end_index": 2
            },
            {
                "title": "Methods", 
                "structure": "2", 
                "physical_index": 3,
                "start_index": 3,
                "end_index": 4,
                "nodes": [
                    {
                        "title": "Subsection", 
                        "structure": "2.1", 
                        "physical_index": 3,
                        "start_index": 3,
                        "end_index": 4
                    }
                ]
            }
        ]
    
    def test_safe_int_conversion_with_valid_int(self):
        """Test safe_int_conversion with valid integer"""
        result = safe_int_conversion(5)
        self.assertEqual(result, 5)
    
    def test_safe_int_conversion_with_valid_string(self):
        """Test safe_int_conversion with valid string"""
        result = safe_int_conversion("5")
        self.assertEqual(result, 5)
    
    def test_safe_int_conversion_with_special_format(self):
        """Test safe_int_conversion with special format"""
        result = safe_int_conversion("<physical_index_5>")
        self.assertEqual(result, 5)
    
    def test_safe_int_conversion_with_none(self):
        """Test safe_int_conversion with None"""
        result = safe_int_conversion(None)
        self.assertIsNone(result)
    
    def test_safe_int_conversion_with_invalid_string(self):
        """Test safe_int_conversion with invalid string"""
        result = safe_int_conversion("invalid")
        self.assertIsNone(result)
    
    def test_add_preface_if_needed_with_no_structure(self):
        """Test add_preface_if_needed with empty structure"""
        result = add_preface_if_needed([], self.mock_context)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Preface")
        self.assertEqual(result[0]["physical_index"], 1)
    
    def test_add_preface_if_needed_with_first_page_not_one(self):
        """Test add_preface_if_needed when first item is not on page 1"""
        structure = [{"title": "Chapter 1", "structure": "1", "physical_index": 3}]
        result = add_preface_if_needed(structure, self.mock_context)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Preface")
        self.assertEqual(result[0]["physical_index"], 1)
        self.assertEqual(result[1]["title"], "Chapter 1")
    
    def test_add_preface_if_needed_with_first_page_one(self):
        """Test add_preface_if_needed when first item is on page 1"""
        structure = [{"title": "Introduction", "structure": "1", "physical_index": 1}]
        result = add_preface_if_needed(structure, self.mock_context)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["title"], "Introduction")
    
    def test_build_tree_structure(self):
        """Test build_tree_structure function"""
        # Add appear_start to mock structure for testing
        structure = [
            {"title": "Introduction", "structure": "1", "physical_index": 1, "appear_start": "yes"},
            {"title": "Methods", "structure": "2", "physical_index": 3, "appear_start": "yes"},
            {"title": "Results", "structure": "3", "physical_index": 5, "appear_start": "yes"}
        ]
        
        result = build_tree_structure(structure, 5, self.mock_context)
        
        # Check that start_index and end_index were added
        self.assertEqual(result[0]["start_index"], 1)
        self.assertEqual(result[0]["end_index"], 2)  # 3-1 = 2
        self.assertEqual(result[1]["start_index"], 3)
        self.assertEqual(result[1]["end_index"], 4)  # 5-1 = 4
        self.assertEqual(result[2]["start_index"], 5)
        self.assertEqual(result[2]["end_index"], 5)  # Total pages
    
    def test_list_to_tree_with_simple_structure(self):
        """Test list_to_tree with simple flat structure"""
        structure = [
            {"title": "Chapter 1", "structure": "1", "start_index": 1, "end_index": 2},
            {"title": "Chapter 2", "structure": "2", "start_index": 3, "end_index": 5}
        ]
        
        result = list_to_tree(structure)
        
        # Should have 2 root nodes
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Chapter 1")
        self.assertEqual(result[1]["title"], "Chapter 2")
    
    def test_list_to_tree_with_hierarchical_structure(self):
        """Test list_to_tree with hierarchical structure"""
        structure = [
            {"title": "Chapter 1", "structure": "1", "start_index": 1, "end_index": 2},
            {"title": "Section 1.1", "structure": "1.1", "start_index": 1, "end_index": 2},
            {"title": "Chapter 2", "structure": "2", "start_index": 3, "end_index": 5}
        ]
        
        result = list_to_tree(structure)
        
        # Should have 2 root nodes
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["title"], "Chapter 1")
        self.assertEqual(result[1]["title"], "Chapter 2")
        
        # First chapter should have one child
        self.assertIn("nodes", result[0])
        self.assertEqual(len(result[0]["nodes"]), 1)
        self.assertEqual(result[0]["nodes"][0]["title"], "Section 1.1")
    
    def test_add_node_ids(self):
        """Test add_node_ids function"""
        structure = [
            {
                "title": "Chapter 1",
                "nodes": [
                    {"title": "Section 1.1"},
                    {"title": "Section 1.2"}
                ]
            },
            {"title": "Chapter 2"}
        ]
        
        add_node_ids(structure)
        
        # Check that node_ids were added
        self.assertEqual(structure[0]["node_id"], "0000")
        self.assertEqual(structure[0]["nodes"][0]["node_id"], "0001")
        self.assertEqual(structure[0]["nodes"][1]["node_id"], "0002")
        self.assertEqual(structure[1]["node_id"], "0003")
    
    def test_add_node_text_recursive(self):
        """Test add_node_text_recursive function"""
        structure = [
            {
                "title": "Chapter 1",
                "start_index": 1,
                "end_index": 2,
                "nodes": [
                    {
                        "title": "Section 1.1",
                        "start_index": 1,
                        "end_index": 1
                    }
                ]
            }
        ]
        
        add_node_text_recursive(structure, self.mock_pages, self.mock_context)
        
        # Check that text was added
        self.assertIn("text", structure[0])
        self.assertIn("text", structure[0]["nodes"][0])
        
        # Check text content
        self.assertEqual(structure[0]["text"], "Page 1 content herePage 2 content here")
        self.assertEqual(structure[0]["nodes"][0]["text"], "Page 1 content here")
    
    def test_remove_node_text(self):
        """Test remove_node_text function"""
        structure = [
            {
                "title": "Chapter 1",
                "text": "Some text",
                "nodes": [
                    {
                        "title": "Section 1.1",
                        "text": "More text"
                    }
                ]
            }
        ]
        
        remove_node_text(structure)
        
        # Check that text was removed
        self.assertNotIn("text", structure[0])
        self.assertNotIn("text", structure[0]["nodes"][0])
    
    def test_apply_enhancements_with_node_ids(self):
        """Test apply_enhancements with node_ids enhancement"""
        structure = [
            {"title": "Chapter 1"},
            {"title": "Chapter 2"}
        ]
        
        enhancements = ["node_ids"]
        result = apply_enhancements(structure, self.mock_pages, enhancements, "gpt-3.5-turbo", self.mock_context)
        
        # Check that node_ids were added
        self.assertIn("node_id", result[0])
        self.assertIn("node_id", result[1])
    
    def test_apply_enhancements_with_node_text(self):
        """Test apply_enhancements with node_text enhancement"""
        structure = [
            {
                "title": "Chapter 1",
                "start_index": 1,
                "end_index": 2
            }
        ]
        
        enhancements = ["node_text"]
        result = apply_enhancements(structure, self.mock_pages, enhancements, "gpt-3.5-turbo", self.mock_context)
        
        # Check that text was added
        self.assertIn("text", result[0])
        self.assertEqual(result[0]["text"], "Page 1 content herePage 2 content here")
    
    def test_count_nodes(self):
        """Test count_nodes function"""
        structure = [
            {
                "title": "Chapter 1",
                "nodes": [
                    {"title": "Section 1.1"},
                    {"title": "Section 1.2"}
                ]
            },
            {"title": "Chapter 2"}
        ]
        
        result = count_nodes(structure)
        self.assertEqual(result, 4)  # 2 chapters + 2 sections
    
    def test_calculate_tree_depth(self):
        """Test calculate_tree_depth function"""
        structure = [
            {
                "title": "Chapter 1",
                "nodes": [
                    {
                        "title": "Section 1.1",
                        "nodes": [
                            {"title": "Subsection 1.1.1"}
                        ]
                    }
                ]
            }
        ]
        
        result = calculate_tree_depth(structure)
        self.assertEqual(result, 3)  # Root -> Section -> Subsection


if __name__ == '__main__':
    unittest.main()
