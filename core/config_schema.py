"""
Configuration schema validation for PageIndex Agent
Provides type-safe configuration validation with sensible defaults
"""

from typing import Dict, Any
from dataclasses import dataclass, field
import re
from core.exceptions import PageIndexError


@dataclass
class GlobalConfig:
    """Global configuration settings"""
    model: str = "gpt-4.1-mini"
    log_dir: str = "./logs"
    session_timeout: int = 3600
    max_tokens_per_call: int = 4000
    retry_attempts: int = 3
    timeout_seconds: int = 30

    def validate(self) -> None:
        """Validate global configuration"""
        if not isinstance(self.model, str) or not self.model.strip():
            raise PageIndexError("Model must be a non-empty string")
        if not isinstance(self.log_dir, str) or not self.log_dir.strip():
            raise PageIndexError("Log directory must be a non-empty string")
        if not isinstance(self.session_timeout, int) or self.session_timeout <= 0:
            raise PageIndexError("Session timeout must be a positive integer")
        if not isinstance(self.max_tokens_per_call, int) or self.max_tokens_per_call < 100:
            raise PageIndexError("Max tokens per call must be >= 100")
        if not isinstance(self.retry_attempts, int) or not 1 <= self.retry_attempts <= 10:
            raise PageIndexError("Retry attempts must be between 1 and 10")
        if not isinstance(self.timeout_seconds, int) or self.timeout_seconds <= 0:
            raise PageIndexError("Timeout seconds must be a positive integer")


@dataclass
class PDFParserConfig:
    """PDF parser configuration"""
    pdf_parser: str = "PyMuPDF"
    max_file_size_mb: int = 100

    def validate(self) -> None:
        """Validate PDF parser configuration"""
        if self.pdf_parser not in ["PyMuPDF", "PyPDF2"]:
            raise PageIndexError("PDF parser must be either 'PyMuPDF' or 'PyPDF2'")
        if not isinstance(self.max_file_size_mb, int) or self.max_file_size_mb <= 0:
            raise PageIndexError("Max file size must be a positive integer")


@dataclass
class TOCDetectorConfig:
    """TOC detector configuration"""
    toc_check_page_num: int = 20

    def validate(self) -> None:
        """Validate TOC detector configuration"""
        if not isinstance(self.toc_check_page_num, int) or self.toc_check_page_num <= 0:
            raise PageIndexError("TOC check page number must be a positive integer")


@dataclass
class StructureExtractorConfig:
    """Structure extractor configuration"""
    max_token_num_each_node: int = 20000
    max_retries: int = 3

    def validate(self) -> None:
        """Validate structure extractor configuration"""
        if not isinstance(self.max_token_num_each_node, int) or not 1000 <= self.max_token_num_each_node <= 50000:
            raise PageIndexError("Max token number each node must be between 1000 and 50000")
        if not isinstance(self.max_retries, int) or not 1 <= self.max_retries <= 5:
            raise PageIndexError("Max retries must be between 1 and 5")


@dataclass
class StructureVerifierConfig:
    """Structure verifier configuration"""
    max_fix_attempts: int = 3
    accuracy_threshold: float = 0.6

    def validate(self) -> None:
        """Validate structure verifier configuration"""
        if not isinstance(self.max_fix_attempts, int) or not 1 <= self.max_fix_attempts <= 5:
            raise PageIndexError("Max fix attempts must be between 1 and 5")
        if not isinstance(self.accuracy_threshold, float) or not 0.0 <= self.accuracy_threshold <= 1.0:
            raise PageIndexError("Accuracy threshold must be between 0.0 and 1.0")


@dataclass
class StructureProcessorConfig:
    """Structure processor configuration"""
    max_page_num_each_node: int = 10
    max_token_num_each_node: int = 20000
    enable_batch_processing: bool = True
    enable_streaming: bool = False
    if_add_node_id: str = "yes"
    if_add_node_summary: str = "no"
    if_add_doc_description: str = "yes"
    if_add_node_text: str = "no"

    def validate(self) -> None:
        """Validate structure processor configuration"""
        if not isinstance(self.max_page_num_each_node, int) or not 1 <= self.max_page_num_each_node <= 50:
            raise PageIndexError("Max page number each node must be between 1 and 50")
        if not isinstance(self.max_token_num_each_node, int) or not 1000 <= self.max_token_num_each_node <= 50000:
            raise PageIndexError("Max token number each node must be between 1000 and 50000")
        
        bool_fields = ["enable_batch_processing", "enable_streaming"]
        for field_name in bool_fields:
            value = getattr(self, field_name)
            if not isinstance(value, bool):
                raise PageIndexError(f"{field_name} must be a boolean")
        
        string_bool_fields = ["if_add_node_id", "if_add_node_summary", "if_add_doc_description", "if_add_node_text"]
        for field_name in string_bool_fields:
            value = getattr(self, field_name)
            if value not in ["yes", "no"]:
                raise PageIndexError(f"{field_name} must be either 'yes' or 'no'")


@dataclass
class PageIndexConfig:
    """Complete PageIndex configuration with validation"""
    global_config: GlobalConfig = field(default_factory=GlobalConfig)
    pdf_parser: PDFParserConfig = field(default_factory=PDFParserConfig)
    toc_detector: TOCDetectorConfig = field(default_factory=TOCDetectorConfig)
    structure_extractor: StructureExtractorConfig = field(default_factory=StructureExtractorConfig)
    structure_verifier: StructureVerifierConfig = field(default_factory=StructureVerifierConfig)
    structure_processor: StructureProcessorConfig = field(default_factory=StructureProcessorConfig)

    def validate(self) -> None:
        """Validate entire configuration"""
        self.global_config.validate()
        self.pdf_parser.validate()
        self.toc_detector.validate()
        self.structure_extractor.validate()
        self.structure_verifier.validate()
        self.structure_processor.validate()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PageIndexConfig':
        """Create validated configuration from dictionary"""
        try:
            global_config = GlobalConfig(**config_dict.get('global', {}))
            pdf_parser = PDFParserConfig(**config_dict.get('pdf_parser', {}))
            toc_detector = TOCDetectorConfig(**config_dict.get('toc_detector', {}))
            structure_extractor = StructureExtractorConfig(**config_dict.get('structure_extractor', {}))
            structure_verifier = StructureVerifierConfig(**config_dict.get('structure_verifier', {}))
            structure_processor = StructureProcessorConfig(**config_dict.get('structure_processor', {}))
            
            config = cls(
                global_config=global_config,
                pdf_parser=pdf_parser,
                toc_detector=toc_detector,
                structure_extractor=structure_extractor,
                structure_verifier=structure_verifier,
                structure_processor=structure_processor
            )
            
            config.validate()
            return config
            
        except TypeError as e:
            raise PageIndexError(f"Configuration validation failed: {str(e)}")
        except Exception as e:
            raise PageIndexError(f"Invalid configuration: {str(e)}")


def validate_config_path(path: str) -> str:
    """Validate configuration file path"""
    if not isinstance(path, str):
        raise PageIndexError("Configuration path must be a string")
    
    # Basic path validation
    if not re.match(r'^[\w\-./]+$', path):
        raise PageIndexError("Configuration path contains invalid characters")
    
    return path


def merge_configs(base_config: Dict[str, Any], user_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base configuration with user overrides"""
    merged = base_config.copy()
    
    def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(merged, user_overrides)
    return merged
