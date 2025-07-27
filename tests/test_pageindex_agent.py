"""
Unit tests for the PageIndexAgent class
"""

import unittest
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.pageindex_agent import PageIndexAgent
from core.context import PageIndexContext
from core.config import PageIndexConfig
from core.config_schema import GlobalConfig


class TestPageIndexAgent(unittest.TestCase):
    """Unit tests for PageIndexAgent functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp(prefix="pageindex_agent_test_"))
        
        # Create a test config
        self.test_config = PageIndexConfig(
            global_config=GlobalConfig(
                model="gpt-3.5-turbo",
                log_dir=str(self.test_dir / "logs")
            )
        )
        
        # Create a test context
        self.test_context = PageIndexContext(config=self.test_config)
        self.test_context.session_id = "test_session"
        self.test_context.pdf_metadata = {
            "pdf_name": "test.pdf",
            "page_count": 10
        }
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('agent.pageindex_agent.openai.OpenAI')
    def test_init(self, mock_openai):
        """Test PageIndexAgent initialization"""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock ConfigManager
        with patch('agent.pageindex_agent.ConfigManager') as mock_config_manager:
            mock_config_manager_instance = MagicMock()
            mock_config_manager.return_value = mock_config_manager_instance
            mock_config_manager_instance.load_config.return_value = self.test_config
            
            # Mock register_tool_functions
            with patch('agent.pageindex_agent.register_tool_functions') as mock_register:
                mock_register.return_value = {}
                
                # Create agent
                agent = PageIndexAgent()
                
                # Assertions
                self.assertEqual(agent.client, mock_client)
                self.assertEqual(agent.config, self.test_config)
                self.assertEqual(agent.tool_functions, {})
                mock_openai.assert_called_once()
                mock_config_manager_instance.load_config.assert_called_once_with(None)
    
    @patch('agent.pageindex_agent.openai.OpenAI')
    def test_create_system_prompt(self, mock_openai):
        """Test _create_system_prompt method"""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock ConfigManager
        with patch('agent.pageindex_agent.ConfigManager') as mock_config_manager:
            mock_config_manager_instance = MagicMock()
            mock_config_manager.return_value = mock_config_manager_instance
            mock_config_manager_instance.load_config.return_value = self.test_config
            
            # Mock register_tool_functions
            with patch('agent.pageindex_agent.register_tool_functions') as mock_register:
                mock_register.return_value = {}
                
                # Create agent
                agent = PageIndexAgent()
                
                # Get system prompt
                prompt = agent._create_system_prompt()
                
                # Assertions
                self.assertIsInstance(prompt, str)
                self.assertIn("PDF document structure extraction agent", prompt)
                self.assertIn("PDF Parser", prompt)
                self.assertIn("TOC Detector", prompt)
                self.assertIn("Structure Extractor", prompt)
                self.assertIn("Structure Verifier", prompt)
                self.assertIn("Structure Processor", prompt)
    
    @patch('agent.pageindex_agent.openai.OpenAI')
    def test_list_sessions_empty(self, mock_openai):
        """Test list_sessions method with no sessions"""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock ConfigManager
        with patch('agent.pageindex_agent.ConfigManager') as mock_config_manager:
            mock_config_manager_instance = MagicMock()
            mock_config_manager.return_value = mock_config_manager_instance
            mock_config_manager_instance.load_config.return_value = self.test_config
            
            # Mock register_tool_functions
            with patch('agent.pageindex_agent.register_tool_functions') as mock_register:
                mock_register.return_value = {}
                
                # Create agent
                agent = PageIndexAgent()
                
                # Get sessions (should be empty)
                sessions = agent.list_sessions()
                
                # Assertions
                self.assertIsInstance(sessions, list)
                self.assertEqual(len(sessions), 0)
    
    @patch('agent.pageindex_agent.openai.OpenAI')
    def test_get_processing_status_not_found(self, mock_openai):
        """Test get_processing_status method with non-existent session"""
        # Mock the OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock ConfigManager
        with patch('agent.pageindex_agent.ConfigManager') as mock_config_manager:
            mock_config_manager_instance = MagicMock()
            mock_config_manager.return_value = mock_config_manager_instance
            mock_config_manager_instance.load_config.return_value = self.test_config
            
            # Mock register_tool_functions
            with patch('agent.pageindex_agent.register_tool_functions') as mock_register:
                mock_register.return_value = {}
                
                # Create agent
                agent = PageIndexAgent()
                
                # Get status for non-existent session
                status = agent.get_processing_status("nonexistent_session")
                
                # Assertions
                self.assertIsInstance(status, dict)
                self.assertEqual(status["status"], "not_found")


if __name__ == '__main__':
    unittest.main()
