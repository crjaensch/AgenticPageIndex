import unittest
from unittest.mock import patch, MagicMock
from ..tools.pdf_parser import pdf_parser_tool
from ..core.context import PageIndexContext
from ..core.config import ConfigManager

class TestPDFParserTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration and context"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.context = PageIndexContext(self.config)
        
    def test_pdf_parser_success(self):
        """Test successful PDF parsing"""
        with patch('pageindex_agent.core.utils.get_page_tokens') as mock_get_tokens, \
             patch('pageindex_agent.core.utils.get_pdf_name') as mock_get_name, \
             patch('pathlib.Path.exists') as mock_exists, \
             patch('pathlib.Path.stat') as mock_stat:
            
            # Setup mocks
            mock_exists.return_value = True
            mock_stat.return_value = MagicMock(st_size=1024)
            mock_get_name.return_value = "test.pdf"
            mock_get_tokens.return_value = [("Page 1 text", 10), ("Page 2 text", 15)]
            
            # Test
            result = pdf_parser_tool(self.context.to_dict(), "test.pdf")
            
            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["confidence"], 1.0)
            self.assertEqual(result["metrics"]["pages_extracted"], 2)
            self.assertEqual(result["metrics"]["total_tokens"], 25)
            
    def test_pdf_parser_file_not_found(self):
        """Test PDF parser with non-existent file"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = pdf_parser_tool(self.context.to_dict(), "nonexistent.pdf")
            
            self.assertFalse(result["success"])
            self.assertEqual(result["confidence"], 0.0)
            self.assertIn("PDF file not found", result["errors"][0])
            self.assertIn("Verify PDF file exists", result["suggestions"][0])
            
    def test_pdf_parser_invalid_file_type(self):
        """Test PDF parser with non-PDF file"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            result = pdf_parser_tool(self.context.to_dict(), "test.txt")
            
            self.assertFalse(result["success"])
            self.assertIn("File is not a PDF", result["errors"][0])
