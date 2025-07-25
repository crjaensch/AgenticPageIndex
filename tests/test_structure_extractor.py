import unittest
from unittest.mock import patch, MagicMock
from tools.structure_extractor import structure_extractor_tool, transform_toc_to_json
from core.context import PageIndexContext
from core.config import ConfigManager

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
        with patch('tools.structure_extractor.extract_with_toc_pages') as mock_extract, \
             patch('tools.structure_extractor.PageIndexContext.load_pages') as mock_load:
            # Setup mocks
            mock_load.return_value = self.mock_pages
            self.context.toc_info = {"found": True, "has_page_numbers": True, "content": "TOC content"}
            mock_extract.return_value = [
                {"structure": "1", "title": "Introduction", "physical_index": 4}
            ]

            result = structure_extractor_tool(self.context.to_dict(), "toc_with_pages")

            # Assertions
            self.assertTrue(result["success"])
            self.assertIn("structure_raw", result["context"])

    def test_structure_extractor_no_toc(self):
        """Test structure extraction without TOC"""
        with patch('tools.structure_extractor.extract_without_toc') as mock_extract, \
             patch('tools.structure_extractor.PageIndexContext.load_pages') as mock_load:
            # Setup mocks
            mock_load.return_value = self.mock_pages
            mock_extract.return_value = [
                {"structure": "1", "title": "Chapter 1", "physical_index": 1}
            ]

            result = structure_extractor_tool(self.context.to_dict(), "no_toc")

            # Assertions
            self.assertTrue(result["success"])
            self.assertIn("structure_raw", result["context"])

    def test_structure_extractor_no_pages(self):
        """Test structure extraction with no pages available"""
        with patch.object(self.context, 'load_pages') as mock_load:
            mock_load.return_value = []
            result = structure_extractor_tool(self.context.to_dict(), "no_toc")
            self.assertFalse(result["success"])
            self.assertIn("No pages data available", result["errors"][0])

    def test_structure_extractor_invalid_strategy(self):
        """Test structure extraction with an invalid strategy"""
        with patch('tools.structure_extractor.PageIndexContext.load_pages') as mock_load:
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
            result = transform_toc_to_json(toc_content, "gpt-4.1-mini")
            
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["title"], "Introduction")
            self.assertEqual(result[1]["title"], "Methods")

if __name__ == '__main__':
    unittest.main()
