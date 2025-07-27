"""
Unit tests for configuration schema validation
"""

import sys
from pathlib import Path
# Add the project root to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
import yaml
from core.config import ConfigManager
from core.config_schema import PageIndexConfig
from core.exceptions import PageIndexError


def test_valid_configuration():
    """Test that valid configuration loads successfully"""
    config_data = {
        "global": {
            "model": "gpt-4.1-mini",
            "log_dir": "./test_logs",
            "session_timeout": 3600,
            "max_tokens_per_call": 4000,
            "retry_attempts": 3,
            "timeout_seconds": 30
        },
        "pdf_parser": {
            "pdf_parser": "PyMuPDF",
            "max_file_size_mb": 100
        },
        "toc_detector": {
            "toc_check_page_num": 20
        },
        "structure_extractor": {
            "max_token_num_each_node": 20000,
            "max_retries": 3
        },
        "structure_verifier": {
            "max_fix_attempts": 3,
            "accuracy_threshold": 0.6
        },
        "structure_processor": {
            "max_page_num_each_node": 10,
            "enable_batch_processing": True,
            "enable_streaming": False,
            "if_add_node_id": "yes",
            "if_add_node_summary": "no",
            "if_add_doc_description": "yes",
            "if_add_node_text": "no"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        assert isinstance(config, PageIndexConfig)
        assert config.global_config.model == "gpt-4.1-mini"
        assert config.pdf_parser.pdf_parser == "PyMuPDF"
        print("✅ Valid configuration test passed")
    finally:
        Path(config_path).unlink()

def test_invalid_model_name():
    """Test that invalid model name raises error"""
    config_data = {
        "global": {
            "model": "",  # Empty string should fail
            "log_dir": "./logs"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config_manager = ConfigManager(config_path)
        try:
            config_manager.load_config()
            assert False, "Should have raised an error"
        except PageIndexError as e:
            assert "Model must be a non-empty string" in str(e)
            print("✅ Invalid model name test passed")
    finally:
        Path(config_path).unlink()

def test_invalid_pdf_parser():
    """Test that invalid PDF parser raises error"""
    config_data = {
        "pdf_parser": {
            "pdf_parser": "InvalidParser"  # Should fail
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config_manager = ConfigManager(config_path)
        try:
            config_manager.load_config()
            assert False, "Should have raised an error"
        except PageIndexError as e:
            assert "PDF parser must be either" in str(e)
            print("✅ Invalid PDF parser test passed")
    finally:
        Path(config_path).unlink()

def test_user_overrides():
    """Test that user overrides are properly applied"""
    base_config = {
        "global": {
            "model": "gpt-3.5-turbo"
        }
    }
    
    user_overrides = {
        "global": {
            "model": "gpt-4.1-mini"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(base_config, f)
        config_path = f.name
    
    try:
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config(user_overrides)
        assert config.global_config.model == "gpt-4.1-mini"
        print("✅ User overrides test passed")
    finally:
        Path(config_path).unlink()

def test_missing_config_file():
    """Test that missing config file uses defaults"""
    config_manager = ConfigManager("/nonexistent/path/config.yaml")
    config = config_manager.load_config()
    assert isinstance(config, PageIndexConfig)
    assert config.global_config.model == "gpt-4.1-mini"  # Default value
    print("✅ Missing config file test passed")


if __name__ == "__main__":
    # Run tests manually for verification
    print("Running configuration validation tests...")
    
    try:
        test_valid_configuration()
        print("✅ Valid configuration test passed")
    except Exception as e:
        print(f"❌ Valid configuration test failed: {e}")
    
    try:
        test_invalid_model_name()
        print("✅ Invalid model name test passed")
    except Exception as e:
        print(f"❌ Invalid model name test failed: {e}")
    
    try:
        test_missing_config_file()
        print("✅ Missing config file test passed")
    except Exception as e:
        print(f"❌ Missing config file test failed: {e}")
    
    print("Configuration validation tests completed!")
