"""
Integration tests for the full PageIndex Agent pipeline
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.pageindex_agent import PageIndexAgent
from core.config import ConfigManager


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the full PageIndex Agent pipeline"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Create a temporary directory for test logs
        self.test_log_dir = Path(tempfile.mkdtemp(prefix="pageindex_test_"))
        self.config_overrides = {
            "global": {
                "log_dir": str(self.test_log_dir)
            }
        }
    
    def tearDown(self):
        """Clean up test files"""
        # Clean up test log directory
        if self.test_log_dir.exists():
            import shutil
            shutil.rmtree(self.test_log_dir)
    
    @patch('agent.pageindex_agent.register_tool_functions')
    @patch('openai.OpenAI')
    @patch('core.utils.get_page_tokens')
    def test_full_pipeline_with_toc_and_page_numbers(self, mock_get_page_tokens, mock_openai, mock_register_tool_functions):
        """Test full pipeline with TOC and page numbers"""
        # Create a temporary PDF file with minimal content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            # Write minimal PDF content
            f.write(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000178 00000 n\ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n283\n%%EOF')
            self.pdf_path = f.name
        
        # Mock get_page_tokens to avoid needing an actual PDF file
        mock_get_page_tokens.return_value = [
            ("Page 1 content", 100),
            ("Page 2 content", 150),
            ("Page 3 content", 200),
            ("Page 4 content", 180),
            ("Page 5 content", 220),
            ("Page 6 content", 190),
            ("Page 7 content", 170),
            ("Page 8 content", 160),
            ("Page 9 content", 140),
            ("Page 10 content", 130)
        ]
        
        # Setup mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock the agent conversation flow
        self._setup_mock_conversation_flow(mock_client)
        
        # Mock tool functions
        mock_pdf_parser = MagicMock()
        mock_toc_detector = MagicMock()
        mock_structure_extractor = MagicMock()
        mock_structure_verifier = MagicMock()
        mock_structure_processor = MagicMock()
        
        # Mock register_tool_functions to return our mocked tool functions
        mock_register_tool_functions.return_value = {
            "pdf_parser": mock_pdf_parser,
            "toc_detector": mock_toc_detector,
            "structure_extractor": mock_structure_extractor,
            "structure_verifier": mock_structure_verifier,
            "structure_processor": mock_structure_processor
        }
        
        # Mock tool function responses with proper context
        # Each mock needs to return the context that would be passed to the next tool
        pdf_parser_context = {
            "pdf_metadata": {
                "pdf_name": "test.pdf",
                "pdf_path": self.pdf_path,
                "total_pages": 10,
                "total_tokens": 5000
            }
        }
        
        toc_detector_context = pdf_parser_context.copy()
        toc_detector_context.update({
            "toc_info": {"found": True, "has_page_numbers": True, "content": "1. Introduction...1\n2. Methods...5"}
        })
        
        structure_extractor_context = toc_detector_context.copy()
        structure_extractor_context.update({
            "structure_raw": [
                {"title": "Introduction", "physical_index": 1, "structure": "1"},
                {"title": "Methods", "physical_index": 5, "structure": "2"}
            ]
        })
        
        structure_verifier_context = structure_extractor_context.copy()
        structure_verifier_context.update({
            "structure_verified": [
                {"title": "Introduction", "physical_index": 1, "structure": "1"},
                {"title": "Methods", "physical_index": 5, "structure": "2"}
            ]
        })
        
        structure_processor_context = structure_verifier_context.copy()
        structure_processor_context.update({
            "structure_final": {
                "title": "Document Title",
                "children": [
                    {"title": "Introduction", "page": 1},
                    {"title": "Methods", "page": 5}
                ]
            }
        })
        
        mock_pdf_parser.return_value = {
            "success": True,
            "confidence": 1.0,
            "metrics": {"pages_extracted": 10, "total_tokens": 5000},
            "context": pdf_parser_context
        }
        
        mock_toc_detector.return_value = {
            "success": True,
            "confidence": 0.9,
            "metrics": {"toc_found": True, "has_page_numbers": True, "toc_pages_count": 2},
            "context": toc_detector_context
        }
        
        mock_structure_extractor.return_value = {
            "success": True,
            "confidence": 0.85,
            "context": structure_extractor_context
        }
        
        mock_structure_verifier.return_value = {
            "success": True,
            "confidence": 0.95,
            "context": structure_verifier_context
        }
        
        mock_structure_processor.return_value = {
            "success": True,
            "context": structure_processor_context
        }
        
        # Initialize agent
        agent = PageIndexAgent(config_overrides=self.config_overrides)
        
        try:
            # Run the pipeline
            result = agent.process_pdf(self.pdf_path)
            
            # Verify result structure
            self.assertIsInstance(result, dict)
            self.assertIn('title', result)
            self.assertIn('children', result)
            
            # Verify tool functions were called
            mock_pdf_parser.assert_called()
            mock_toc_detector.assert_called()
            mock_structure_extractor.assert_called()
            mock_structure_verifier.assert_called()
            mock_structure_processor.assert_called()
            
        finally:
            # Clean up
            Path(self.pdf_path).unlink()
    
    def _setup_mock_conversation_flow(self, mock_client):
        """Setup mock conversation flow for the agent"""
        # Mock responses for each step in the pipeline
        mock_responses = [
            # Step 1: PDF Parser
            self._create_mock_response([
                self._create_mock_tool_call(
                    "call_1", 
                    "pdf_parser", 
                    {"pdf_path": self.pdf_path}
                )
            ], "PDF parsing completed"),
            
            # Step 2: TOC Detector
            self._create_mock_response([
                self._create_mock_tool_call(
                    "call_2", 
                    "toc_detector", 
                    {}
                )
            ], "TOC detection completed"),
            
            # Step 3: Structure Extractor (with TOC and page numbers)
            self._create_mock_response([
                self._create_mock_tool_call(
                    "call_3", 
                    "structure_extractor", 
                    {"strategy": "toc_with_pages"}
                )
            ], "Structure extraction completed"),
            
            # Step 4: Structure Verifier
            self._create_mock_response([
                self._create_mock_tool_call(
                    "call_4", 
                    "structure_verifier", 
                    {}
                )
            ], "Structure verification completed"),
            
            # Step 5: Structure Processor
            self._create_mock_response([
                self._create_mock_tool_call(
                    "call_5", 
                    "structure_processor", 
                    {}
                )
            ], "Final structure processing completed"),
            
            # Final response
            self._create_mock_response(None, "Processing completed successfully")
        ]
        
        mock_client.chat.completions.create.side_effect = mock_responses
    
    def _create_mock_response(self, tool_calls, content):
        """Create a mock response for the OpenAI client"""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        mock_message.tool_calls = tool_calls
        mock_message.content = content
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        return mock_response
    
    def _create_mock_tool_call(self, id, function_name, arguments):
        """Create a mock tool call with proper structure"""
        mock_tool_call = MagicMock()
        mock_function = MagicMock()
        
        mock_function.name = function_name
        mock_function.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
        mock_tool_call.function = mock_function
        mock_tool_call.id = id
        mock_tool_call.type = "function"
        
        return mock_tool_call

if __name__ == '__main__':
    unittest.main()
