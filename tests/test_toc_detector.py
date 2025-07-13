import unittest
from unittest.mock import patch, MagicMock
from tools.toc_detector import toc_detector_tool, detect_toc_single_page
from core.context import PageIndexContext
from core.config import ConfigManager

class TestTOCDetectorTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration and context"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.context = PageIndexContext(self.config)
        
        # Mock pages data
        self.mock_pages = [
            ("Regular content page", 50),
            ("Table of Contents\n1. Introduction ... 1\n2. Methods ... 5", 30),
            ("3. Results ... 10\n4. Conclusion ... 15", 25),
            ("Chapter 1: Introduction", 40)
        ]
        
    def test_toc_detector_with_toc_found(self):
        """Test TOC detector when TOC is found"""
        with patch('tools.toc_detector.PageIndexContext.load_pages') as mock_load, \
             patch('tools.toc_detector.detect_toc_single_page') as mock_detect, \
             patch('tools.toc_detector.extract_toc_content') as mock_extract, \
             patch('tools.toc_detector.detect_page_numbers_in_toc') as mock_page_nums:  # Mock the client to prevent instantiation

            # Setup mocks
            mock_load.return_value = self.mock_pages
            mock_detect.side_effect = ['no', 'yes', 'yes', 'no']  # TOC on pages 1 and 2
            mock_extract.return_value = {
                "content": "1. Introduction ... 1\n2. Methods ... 5",
                "has_page_numbers": True
            }
            mock_page_nums.return_value = True

            result = toc_detector_tool(self.context.to_dict())

            # Assertions
            self.assertTrue(result["success"])
            self.assertTrue(result["metrics"]["toc_found"])
            self.assertTrue(result["metrics"]["has_page_numbers"])
            self.assertEqual(result["metrics"]["toc_pages_count"], 2)

    def test_toc_detector_no_toc_found(self):
        """Test TOC detector when no TOC is found"""
        with patch.object(self.context, 'load_pages') as mock_load, \
             patch('tools.toc_detector.detect_toc_single_page') as mock_detect:

            # Setup mocks
            mock_load.return_value = self.mock_pages
            mock_detect.return_value = 'no'  # No TOC found

            result = toc_detector_tool(self.context.to_dict())

            # Assertions
            self.assertFalse(result["success"])
            self.assertFalse(result["metrics"]["toc_found"])
            self.assertEqual(result["metrics"]["toc_pages_count"], 0)

    def test_detect_toc_single_page(self):
        """Test single page TOC detection"""
        with patch('openai.OpenAI') as mock_openai:
            # Setup mock response
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_response = MagicMock()
            mock_response.choices[0].message.content = '{"toc_detected": "yes"}'
            mock_client.chat.completions.create.return_value = mock_response
            
            result = detect_toc_single_page("Table of Contents\n1. Introduction", "gpt-4o-mini")
            
            self.assertEqual(result, "yes")

if __name__ == '__main__':
    unittest.main()
