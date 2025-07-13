import unittest
from unittest.mock import patch, MagicMock
from ..agent.pageindex_agent import PageIndexAgent

class TestPageIndexIntegration(unittest.TestCase):
    """Integration tests for the complete PageIndex Agent workflow"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "global": {"model": "gpt-4o-mini", "log_dir": "./test_logs"},
            "pdf_parser": {"pdf_parser": "PyMuPDF"},
            "toc_detector": {"toc_check_page_num": 5},
            "structure_extractor": {"max_token_num_each_node": 10000},
            "structure_verifier": {"accuracy_threshold": 0.6, "max_fix_attempts": 2},
            "structure_processor": {
                "max_page_num_each_node": 5,
                "if_add_node_id": "yes",
                "if_add_doc_description": "yes"
            }
        }
        
        # Create test agent
        self.agent = PageIndexAgent(
            api_key="test_key",
            config_overrides=self.test_config
        )
        
    @patch('openai.OpenAI')
    @patch('pageindex_agent.core.utils.get_page_tokens')
    @patch('pageindex_agent.core.utils.get_pdf_name')
    @patch('pathlib.Path.exists')
    def test_complete_workflow_with_toc(self, mock_exists, mock_get_name, mock_get_tokens, mock_openai):
        """Test complete workflow with a document that has TOC"""
        
        # Setup file system mocks
        mock_exists.return_value = True
        mock_get_name.return_value = "test_document.pdf"
        mock_get_tokens.return_value = [
            ("Regular content", 50),
            ("Table of Contents\n1. Introduction ... 1\n2. Methods ... 5", 30),
            ("Introduction content here", 40),
            ("Methods section content", 45),
            ("Results and conclusions", 50)
        ]
        
        # Setup OpenAI client mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock agent conversation - the agent will make tool calls
        agent_responses = [
            # First response: agent decides to parse PDF
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=[MagicMock(
                    function=MagicMock(name="pdf_parser", arguments='{"pdf_path": "test.pdf"}'),
                    id="call_1"
                )],
                to_dict=lambda: {"role": "assistant", "content": None}
            ))]),
            # Second response: agent detects TOC
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=[MagicMock(
                    function=MagicMock(name="toc_detector", arguments='{}'),
                    id="call_2"
                )],
                to_dict=lambda: {"role": "assistant", "content": None}
            ))]),
            # Third response: agent extracts structure
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=[MagicMock(
                    function=MagicMock(name="structure_extractor", arguments='{"strategy": "toc_with_pages"}'),
                    id="call_3"
                )],
                to_dict=lambda: {"role": "assistant", "content": None}
            ))]),
            # Fourth response: agent verifies structure
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=[MagicMock(
                    function=MagicMock(name="structure_verifier", arguments='{}'),
                    id="call_4"
                )],
                to_dict=lambda: {"role": "assistant", "content": None}
            ))]),
            # Fifth response: agent processes final structure
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=[MagicMock(
                    function=MagicMock(name="structure_processor", arguments='{}'),
                    id="call_5"
                )],
                to_dict=lambda: {"role": "assistant", "content": None}
            ))]),
            # Final response: agent finishes
            MagicMock(choices=[MagicMock(message=MagicMock(
                tool_calls=None,
                to_dict=lambda: {"role": "assistant", "content": "Processing completed successfully!"}
            ))])
        ]
        
        mock_client.chat.completions.create.side_effect = agent_responses
        
        # Mock all the LLM calls within tools
        with patch('pageindex_agent.tools.toc_detector.detect_toc_single_page') as mock_detect_toc, \
             patch('pageindex_agent.tools.toc_detector.detect_page_numbers_in_toc') as mock_detect_pages, \
             patch('pageindex_agent.tools.structure_extractor.transform_toc_to_json') as mock_transform, \
             patch('pageindex_agent.tools.structure_verifier.verify_structure_accuracy') as mock_verify, \
             patch('pageindex_agent.tools.structure_processor.generate_document_description') as mock_doc_desc:
            
            # Setup tool mocks
            mock_detect_toc.side_effect = ['no', 'yes', 'no', 'no', 'no']  # TOC on page 1
            mock_detect_pages.return_value = True
            mock_transform.return_value = [
                {"structure": "1", "title": "Introduction", "page": 1},
                {"structure": "2", "title": "Methods", "page": 5}
            ]
            
            async def mock_verify_accuracy(*args):
                return 1.0, []  # Perfect accuracy
            
            mock_verify.return_value = mock_verify_accuracy()
            mock_doc_desc.return_value = "A test document about research methods"
            
            # Execute test
            result = self.agent.process_pdf("test.pdf")
            
            # Verify results
            self.assertIsInstance(result, dict)
            self.assertIn("doc_name", result)
            self.assertIn("structure", result)
            self.assertEqual(result["doc_name"], "test_document.pdf")
            
    def test_error_handling_and_recovery(self):
        """Test error handling and diagnostic information saving"""
        
        # Mock a tool that fails
        with patch('pageindex_agent.tools.pdf_parser.pdf_parser_tool') as mock_parser, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_exists.return_value = False  # File doesn't exist
            mock_parser.return_value = {
                "success": False,
                "errors": ["PDF file not found"],
                "suggestions": ["Verify PDF file exists"],
                "context": {}
            }
            
            # Test should raise error and save diagnostics
            with self.assertRaises(Exception):
                self.agent.process_pdf("nonexistent.pdf")
                
    def test_session_management(self):
        """Test session status and listing functionality"""
        
        # Test with no sessions
        sessions = self.agent.list_sessions()
        self.assertIsInstance(sessions, list)
        
        # Test status check for non-existent session
        status = self.agent.get_processing_status("nonexistent_session")
        self.assertEqual(status["status"], "not_found")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
