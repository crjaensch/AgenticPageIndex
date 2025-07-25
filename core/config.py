from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml
from pathlib import Path
from core.config_schema import PageIndexConfig, merge_configs
from core.exceptions import PageIndexError

@dataclass
class GlobalConfig:
    model: str = "gpt-4.1-mini"
    log_dir: str = "./logs"
    session_timeout: int = 3600

@dataclass  
class PDFParserConfig:
    pdf_parser: str = "PyMuPDF"  # or "PyPDF2"
    
@dataclass
class TOCDetectorConfig:
    toc_check_page_num: int = 20
    
@dataclass
class StructureExtractorConfig:
    max_token_num_each_node: int = 20000
    max_retries: int = 3
    
@dataclass
class StructureVerifierConfig:
    max_fix_attempts: int = 3
    accuracy_threshold: float = 0.6
    
@dataclass
class StructureProcessorConfig:
    max_page_num_each_node: int = 10
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "no" 
    if_add_doc_description: str = "yes"
    if_add_node_text: str = "no"

class ConfigManager:
    def __init__(self, config_path: str = None):
        self.config_path = config_path or Path(__file__).parent.parent / "config.yaml"
    
    def load_config(self, user_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load configuration with user overrides and validation"""
        # Load default config
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
        else:
            base_config = {}
            
        # Apply user overrides
        if user_overrides:
            base_config = merge_configs(base_config, user_overrides)
        
        # Validate configuration using schema
        try:
            validated_config = PageIndexConfig.from_dict(base_config)
            return validated_config
        except Exception as e:
            raise PageIndexError(f"Configuration validation failed: {str(e)}")
    
    def migrate_legacy_config(self, legacy_config_path: str, output_path: str = None):
        """Migrate legacy config.yaml to new hierarchical structure"""
        with open(legacy_config_path, 'r') as f:
            legacy_config = yaml.safe_load(f) or {}
        
        # Map legacy keys to new structure
        new_config = {
            "global": {
                "model": legacy_config.get("model", "gpt-4.1-mini")
            },
            "toc_detector": {
                "toc_check_page_num": legacy_config.get("toc_check_page_num", 20)
            },
            "structure_processor": {
                "max_page_num_each_node": legacy_config.get("max_page_num_each_node", 10),
                "if_add_node_id": legacy_config.get("if_add_node_id", "yes"),
                "if_add_node_summary": legacy_config.get("if_add_node_summary", "no"),
                "if_add_doc_description": legacy_config.get("if_add_doc_description", "yes"),
                "if_add_node_text": legacy_config.get("if_add_node_text", "no")
            },
            "structure_extractor": {
                "max_token_num_each_node": legacy_config.get("max_token_num_each_node", 20000)
            }
        }
        
        output_file = output_path or str(Path(legacy_config_path).with_name("config_new.yaml"))
        with open(output_file, 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False, indent=2)
        
        return output_file
