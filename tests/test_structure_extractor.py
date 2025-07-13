import unittest
from unittest.mock import patch, MagicMock
from ..tools.structure_extractor import structure_extractor_tool, transform_toc_to_json
from ..core.context import PageIndexContext
from ..core.config import ConfigManager

class TestStructureExtractorTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration and context"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.context = PageIndexContext(self.config)
        
        # Mock TOC info
        self.context.toc_info = {
            "found": True,
            "pages": [1, 2],
            "content": "1. Introduction ... 1\n2. Methods ... 5\n3. Results ... 10",
            "has_page_numbers": True
        }
        
        # Mock pages data
        self.mock_pages = [
            ("Page 1 content", 50),
            ("Table of Contents\n1. Introduction ... 1", 30),
            ("2. Methods ... 5\n3. Results ... 10", 25),
            ("Introduction content here", 40),
            ("Methods section content", 45)
        ]
        
    def test_structure_extractor_toc_with_pages(self):
        """Test structure extraction with TOC containing page numbers"""
        with patch.object(self.context, 'load_pages') as mock_load, \
             patch('pageindex_agent.tools.structure_extractor.transform_toc_to_json') as mock_transform, \
             patch('pageindex_agent.tools.structure_extractor.extract_toc_physical_indices') as mock_extract, \
             patch('pageindex_agent.tools.structure_extractor.calculate_page_offset') as mock_offset, \
             patch('pageindex_agent.tools.structure_extractor.apply_page_offset') as mock_apply:
            
            # Setup mocks
            mock_load.return_value = self.mock_pages
            mock_transform.return_value = [
                {"structure": "1", "title": "Introduction", "page": 1},
                {"structure": "2", "title": "Methods", "page": 5}
            ]
            mock_extract.return_value = [
                {"structure": "1", "title": "Introduction", "physical_index": 4}
            ]
            mock_offset.return_value = 3
            mock_apply.return_value = [
                {"structure": "1", "title": "Introduction", "physical_index": 4},
                {"structure": "2", "title": "Methods", "physical_index": 8}
            ]
            
            result = structure_extractor_tool(self.context.to_dict(), "toc_with_pages")
            
            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["confidence"], 0.9)
            self.assertEqual(result["metrics"]["strategy_used"], "toc_with_pages")
            self.assertEqual(result["metrics"]["items_extracted"], 2)
            
    def test_structure_extractor_no_toc(self):
        """Test structure extraction without TOC"""
        self.context.toc_info = {"found": False}
        
        with patch.object(self.context, 'load_pages') as mock_load, \
             patch('pageindex_agent.tools.structure_extractor.page_list_to_group_text') as mock_group, \
             patch('pageindex_agent.tools.structure_extractor.generate_structure_from_content') as mock_generate, \
             patch('pageindex_agent.tools.structure_extractor.convert_physical_index_to_int') as mock_convert:
            
            # Setup mocks
            mock_load.return_value = self.mock_pages
            mock_group.return_value = ["<physical_index_1>Page 1 content<physical_index_1>"]
            mock_generate.return_value = [
                {"structure": "1", "title": "Chapter 1", "physical_index": "<physical_index_1>"}
            ]
            mock_convert.return_value = [
                {"structure": "1", "title": "Chapter 1", "physical_index": 1}
            ]
            
            result = structure_extractor_tool(self.context.to_dict(), "no_toc")
            
            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["confidence"], 0.6)
            self.assertEqual(result["metrics"]["strategy_used"], "no_toc")
            
    def test_structure_extractor_invalid_strategy(self):
        """Test structure extractor with invalid strategy"""
        with patch.object(self.context, 'load_pages') as mock_load:
            mock_load.return_value = self.mock_pages
            
            result = structure_extractor_tool(self.context.to_dict(), "invalid_strategy")
            
            self.assertFalse(result["success"])
            self.assertIn("Unknown extraction strategy", result["errors"][0])
            
    def test_transform_toc_to_json(self):
        """Test TOC transformation to JSON"""
        with patch('openai.OpenAI') as mock_openai:
            # Setup mock response
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '''
            {
                "table_of_contents": [
                    {"structure": "1", "title": "Introduction", "page": 1},
                    {"structure": "2", "title": "Methods", "page": 5}
                ]
            }
            '''
            mock_client.chat.completions.create.return_value = mock_response
            
            toc_content = "1. Introduction ... 1\n2. Methods ... 5"
            result = transform_toc_to_json(toc_content, "gpt-4o-mini")
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["title"], "Introduction")
            self.assertEqual(result[1]["title"], "Methods")
