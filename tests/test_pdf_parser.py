import unittest
from unittest.mock import patch, MagicMock
from tools.pdf_parser import pdf_parser_tool
from core.context import PageIndexContext
from core.config import ConfigManager
from core.exceptions import PageIndexToolError

class TestPDFParserTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration and context"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.context = PageIndexContext(self.config)
        
    @patch('tools.pdf_parser.get_page_tokens')
    @patch('tools.pdf_parser.get_pdf_name')
    @patch('pathlib.Path.exists')
    def test_pdf_parser_success(self, mock_exists, mock_get_name, mock_get_tokens):
        """Test successful PDF parsing"""
        with patch('builtins.open'), patch('tools.pdf_parser.open') as mock_open:
            mock_doc = MagicMock()
            mock_page = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_doc
            mock_doc.__len__.return_value = 2
            mock_doc.load_page.return_value = mock_page
            
            # Setup mocks
            mock_exists.return_value = True
            mock_get_name.return_value = "test.pdf"
            mock_get_tokens.return_value = [("page 1 text", 10), ("page 2 text", 15)]
            mock_page.get_text.side_effect = ["Page 1 text", "Page 2 text"]
            
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
            
            with self.assertRaises(PageIndexToolError) as cm:
                pdf_parser_tool(self.context.to_dict(), "nonexistent.pdf")
            self.assertIn("PDF file not found", str(cm.exception))
            
    def test_pdf_parser_invalid_file_type(self):
        """Test PDF parser with non-PDF file"""
        with self.assertRaises(PageIndexToolError) as cm:
            pdf_parser_tool(self.context.to_dict(), "test.txt")
        self.assertIn("PDF file not found: test.txt", str(cm.exception))
