import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from core.config_schema import PageIndexConfig

@dataclass
class PageIndexContext:
    """Context object that carries state through the PageIndex processing pipeline"""
    
    session_id: str
    config: PageIndexConfig
    pdf_metadata: Dict[str, Any] 
    pages_file: Optional[str]  # Path to serialized pages data
    toc_info: Dict[str, Any]
    structure_raw: List[Dict[str, Any]]
    structure_verified: List[Dict[str, Any]]
    structure_final: Dict[str, Any]
    processing_log: List[Dict[str, Any]]
    current_step: str
    
    def __init__(self, config: PageIndexConfig):
        self.session_id = str(uuid.uuid4())
        self.config = config
        self.pdf_metadata = {}
        self.pages_file = None
        self.toc_info = {}
        self.structure_raw = []
        self.structure_verified = []
        self.structure_final = {}
        self.processing_log = []
        self.current_step = "initialized"
    
    def log_step(self, tool_name: str, status: str, details: Dict[str, Any] = None):
        """Add processing step to log"""
        step = {
            "tool": tool_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        self.processing_log.append(step)
        self.current_step = f"{tool_name}_{status}"
    
    def save_pages(self, pages: List[tuple], log_dir: Path):
        """Save pages data to file and store reference"""
        pages_path = log_dir / f"{self.session_id}_pages.json"
        with open(pages_path, 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2)
        self.pages_file = str(pages_path)
    
    def load_pages(self) -> List[tuple]:
        """Load pages data from file"""
        if not self.pages_file:
            return []
        with open(self.pages_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_checkpoint(self, log_dir: Path, include_pages: bool = False):
        """Save current context state for diagnostics"""
        checkpoint_path = log_dir / f"{self.session_id}_checkpoint.json"
        context_dict = asdict(self)
        
        # Optionally include pages data in checkpoint for debugging
        if include_pages and self.pages_file:
            context_dict['pages_data'] = self.load_pages()
            
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(context_dict, f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for Agent SDK (excluding large data)"""
        return {
            "session_id": self.session_id,
            "config": asdict(self.config),
            "pdf_metadata": self.pdf_metadata,
            "pages_file": self.pages_file,
            "toc_info": self.toc_info,
            "structure_raw": self.structure_raw,
            "structure_verified": self.structure_verified,
            "structure_final": self.structure_final,
            "processing_log": self.processing_log[-5:],  # Last 5 steps only
            "current_step": self.current_step
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageIndexContext':
        """Create context from dictionary"""
        from core.config_schema import PageIndexConfig
        
        # Handle both old dict format and new object format
        config_data = data.get('config', {})
        if isinstance(config_data, dict):
            config = PageIndexConfig.from_dict(config_data)
        else:
            config = config_data
            
        context = cls(config)
        
        # Restore state from dictionary
        context.session_id = data.get('session_id', str(uuid.uuid4()))
        context.pdf_metadata = data.get('pdf_metadata', {})
        context.pages_file = data.get('pages_file')
        context.toc_info = data.get('toc_info', {})
        context.structure_raw = data.get('structure_raw', [])
        context.structure_verified = data.get('structure_verified', [])
        context.structure_final = data.get('structure_final', {})
        context.processing_log = data.get('processing_log', [])
        context.current_step = data.get('current_step', "initialized")
        
        return context
