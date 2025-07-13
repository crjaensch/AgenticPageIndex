import unittest
import asyncio
from unittest.mock import patch, MagicMock
from ..tools.structure_verifier import structure_verifier_tool, check_title_on_page
from ..core.context import PageIndexContext
from ..core.config import ConfigManager

class TestStructureVerifierTool(unittest.TestCase):
    
    def setUp(self):
        """Set up test configuration and context"""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        self.context = PageIndexContext(self.config)
        
        # Mock structure data
        self.context.structure_raw = [
            {"title": "Introduction", "physical_index": 1, "structure": "1"},
            {"title": "Methods", "physical_index": 3, "structure": "2"},
            {"title": "Results", "physical_index": 5, "structure": "3"}
        ]
        
        # Mock pages data
        self.mock_pages = [
            ("Introduction section content", 50),
            ("Some content here", 30),
            ("Methods section starts here", 40),
            ("More methods content", 35),
            ("Results and analysis", 45)
        ]
        
    def test_structure_verifier_high_accuracy(self):
        """Test structure verifier with high accuracy"""
        with patch.object(self.context, 'load_pages') as mock_load, \
             patch('pageindex_agent.tools.structure_verifier.verify_structure_accuracy') as mock_verify:
            
            # Setup mocks
            mock_load.return_value = self.mock_pages
            
            # Mock async function
            async def mock_verify_func(*args):
                return 1.0, []  # Perfect accuracy, no errors
            
            mock_verify.return_value = mock_verify_func()
            
            result = structure_verifier_tool(self.context.to_dict())
            
            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["confidence"], 1.0)
            self.assertEqual(result["metrics"]["accuracy"], 1.0)
            self.assertEqual(result["metrics"]["incorrect_items"], 0)
            
    def test_structure_verifier_with_errors(self):
        """Test structure verifier with some errors that need fixing"""
        with patch.object(self.context, 'load_pages') as mock_load, \
             patch('pageindex_agent.tools.structure_verifier.verify_structure_accuracy') as mock_verify, \
             patch('pageindex_agent.tools.structure_verifier.fix_structure_errors') as mock_fix:
            
            # Setup mocks
            mock_load.return_value = self.mock_pages
            
            # Mock verification with errors
            async def mock_verify_func(*args):
                return 0.7, [{"list_index": 1, "title": "Methods", "physical_index": 3}]
            
            mock_verify.return_value = mock_verify_func()
            
            # Mock fixing
            async def mock_fix_func(*args):
                fixed_structure = self.context.structure_raw.copy()
                fixed_structure[1]["physical_index"] = 4  # Fixed index
                return fixed_structure, []  # No remaining errors
            
            mock_fix.return_value = mock_fix_func()
            
            result = structure_verifier_tool(self.context.to_dict())
            
            # Assertions
            self.assertTrue(result["success"])
            self.assertEqual(result["confidence"], 0.8)  # Good accuracy after fixing
            self.assertEqual(result["metrics"]["incorrect_items"], 1)
            
    def test_check_title_on_page(self):
        """Test individual title verification on page"""
        with patch('openai.AsyncOpenAI') as mock_openai:
            # Setup mock response
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            
            async def mock_create(*args, **kwargs):
                mock_response = MagicMock()
                mock_response.choices[0].message.content = '{"answer": "yes"}'
                return mock_response
            
            mock_client.chat.completions.create = mock_create
            
            item = {"title": "Introduction", "physical_index": 1}
            
            # Run async test
            result = asyncio.run(check_title_on_page(item, self.mock_pages, "gpt-4o-mini"))
            
            self.assertEqual(result, "yes")
