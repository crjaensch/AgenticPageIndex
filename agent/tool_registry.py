from typing import Dict, Callable
from tools.pdf_parser import pdf_parser_tool
from tools.toc_detector import toc_detector_tool
from tools.structure_extractor import structure_extractor_tool
from tools.structure_verifier import structure_verifier_tool
from tools.structure_processor import structure_processor_tool

# OpenAI Agent SDK Tool Definitions
PAGEINDEX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pdf_parser",
            "description": "Extract text, metadata and token counts from PDF document",
            "parameters": {
                "type": "object", 
                "properties": {
                    "context": {
                        "type": "object",
                        "description": "Current PageIndex context"
                    },
                    "pdf_path": {
                        "type": "string", 
                        "description": "Path to PDF file to process"
                    }
                },
                "required": ["context", "pdf_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "toc_detector", 
            "description": "Detect table of contents in document and analyze structure",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "description": "Current PageIndex context with pages data"
                    }
                },
                "required": ["context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "structure_extractor",
            "description": "Extract document hierarchy using specified strategy. STRATEGY SELECTION: Use 'toc_with_pages' only if context.toc_info.found=true AND context.toc_info.has_page_numbers=true. Use 'toc_no_pages' only if context.toc_info.found=true AND context.toc_info.has_page_numbers=false. Use 'no_toc' if context.toc_info.found=false OR as fallback when other strategies fail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "description": "Current PageIndex context with pages and toc_info"
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["toc_with_pages", "toc_no_pages", "no_toc"],
                        "description": "Extraction strategy to use"
                    }
                },
                "required": ["context", "strategy"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "structure_verifier",
            "description": "Verify structure accuracy and fix errors",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "description": "Current PageIndex context with structure_raw"
                    }
                },
                "required": ["context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "structure_processor",
            "description": "Generate final tree structure with optional enhancements",
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "object",
                        "description": "Current PageIndex context with structure_verified"
                    },
                    "enhancements": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["node_ids", "summaries", "doc_description", "node_text"]
                        },
                        "description": "List of enhancements to apply"
                    }
                },
                "required": ["context"]
            }
        }
    }
]

def register_tool_functions() -> Dict[str, Callable]:
    """Register actual tool functions for Agent SDK"""
    return {
        "pdf_parser": pdf_parser_tool,
        "toc_detector": toc_detector_tool,
        "structure_extractor": structure_extractor_tool,
        "structure_verifier": structure_verifier_tool,
        "structure_processor": structure_processor_tool
    }
