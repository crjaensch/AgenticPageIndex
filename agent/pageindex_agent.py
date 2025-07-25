import json
import os
from pathlib import Path
from typing import Dict, Any, List
import openai
from core.context import PageIndexContext
from core.config import ConfigManager
from core.exceptions import PageIndexError, PageIndexToolError
from agent.tool_registry import PAGEINDEX_TOOLS, register_tool_functions

class PageIndexAgent:
    """
    Main PageIndex Agent that orchestrates PDF document structure extraction
    using OpenAI Agent SDK and dynamic tool selection
    """
    
    def __init__(self, api_key: str = None, config_overrides: Dict[str, Any] = None, verbose: bool = False):
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config(config_overrides)
        self.tool_functions = register_tool_functions()
        self.verbose = verbose
        
        # Setup logging directory
        log_dir = Path(self.config.global_config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main entry point for PDF structure extraction
        
        Args:
            pdf_path: Path to PDF file to process
            
        Returns:
            Final processed structure with metadata
        """
        
        try:
            # Initialize context
            context = PageIndexContext(self.config)
            
            # Create system prompt for agent orchestration
            system_prompt = self._create_system_prompt()
            
            # Start agent conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Process PDF document: {pdf_path}"}
            ]
            
            # Agent execution loop
            max_iterations = 20  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                
                response = self.client.chat.completions.create(
                    model=self.config.global_config.model,
                    messages=messages,
                    tools=PAGEINDEX_TOOLS,
                    tool_choice="auto",
                    temperature=0
                )
                
                message = response.choices[0].message
                messages.append(message.model_dump())
                
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        if self.verbose:
                            print(f"[Agent] Calling tool: {function_name}")
                        
                        # Inject current context into tool call
                        function_args["context"] = context.to_dict()
                        
                        # Execute tool
                        try:
                            tool_function = self.tool_functions[function_name]
                            result = tool_function(**function_args)
                            
                            # Update context from tool result
                            if result["success"]:
                                context = PageIndexContext.from_dict(result["context"])
                            
                            if self.verbose:
                                if result["success"]:
                                    print(f"[Agent] Tool {function_name} completed successfully")
                                else:
                                    print(f"[Agent] Tool {function_name} failed")
                            
                            # Add tool response to conversation
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            
                            # Check for tool failures
                            if not result["success"]:
                                error_msg = f"Tool {function_name} failed: {result.get('errors', [])}"
                                if result.get("suggestions"):
                                    error_msg += f"\nSuggestions: {result['suggestions']}"
                                raise PageIndexToolError(error_msg)
                                
                        except Exception as e:
                            # Handle tool execution errors
                            error_result = {
                                "success": False,
                                "error": str(e),
                                "context": context.to_dict()
                            }
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(error_result)
                            })
                            
                            # Save failure state for diagnostics
                            log_dir = Path(self.config.global_config.log_dir) / context.session_id
                            log_dir.mkdir(parents=True, exist_ok=True)
                            context.log_step("agent", "tool_failed", {
                                "tool": function_name,
                                "error": str(e)
                            })
                            context.save_checkpoint(log_dir, include_pages=True)
                            
                            raise PageIndexError(f"Tool execution failed: {str(e)}")
                else:
                    # No more tool calls, agent finished
                    break
            
            if iteration >= max_iterations:
                raise PageIndexError("Maximum iterations reached, processing may be incomplete")
            
            # Return final result
            if context.structure_final:
                return context.structure_final
            else:
                raise PageIndexError("Processing completed but no final structure was generated")
                
        except Exception as e:
            # Ensure diagnostic information is saved
            if 'context' in locals():
                log_dir = Path(self.config.global_config.log_dir) / context.session_id
                log_dir.mkdir(parents=True, exist_ok=True)
                context.log_step("agent", "failed", {"error": str(e)})
                context.save_checkpoint(log_dir, include_pages=True)
                
                print(f"Error during processing. Diagnostic information saved to: {log_dir}")
            
            raise
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for agent orchestration"""
        return """
You are a PDF document structure extraction agent. Your task is to process PDF documents and extract their hierarchical structure using the following workflow:

1. **PDF Parser**: Always start by parsing the PDF document to extract text and metadata
2. **TOC Detector**: Detect if the document has a table of contents and analyze its structure
3. **Structure Extractor**: Extract the document hierarchy using the best strategy:
   - MANDATORY: Check context.toc_info state before strategy selection
   - If context.toc_info.found=true AND context.toc_info.has_page_numbers=true → use "toc_with_pages"
   - If context.toc_info.found=true AND context.toc_info.has_page_numbers=false → use "toc_no_pages"
   - If context.toc_info.found=false → use "no_toc"
   - FALLBACK: If any strategy fails due to invalid prerequisites → automatically try "no_toc"
   - NEVER retry the same strategy twice - implement progressive fallback
4. **Structure Verifier**: Verify the extracted structure and fix any errors
5. **Structure Processor**: Generate the final tree structure with requested enhancements

**Important Guidelines:**
- Use tools sequentially - each tool depends on the output of previous tools
- Monitor confidence scores and implement fallback strategies
- Always handle tool failures gracefully with appropriate error messages
- Save intermediate state for diagnostic purposes
- The context object carries all state between tool calls

**Strategy Selection Logic:**
- High confidence (>0.8): Proceed to next step
- Medium confidence (0.6-0.8): Proceed but may need verification/fixing
- Low confidence (<0.6): Try fallback strategy

**Error Handling:**
- Tool failures should trigger appropriate recovery strategies
- Save diagnostic information for manual inspection
- Provide clear error messages and recovery suggestions

Start by parsing the PDF document provided by the user.
"""
    
    def get_processing_status(self, session_id: str) -> Dict[str, Any]:
        """Get processing status for a session"""
        log_dir = Path(self.config.global_config.log_dir) / session_id
        checkpoint_file = log_dir / f"{session_id}_checkpoint.json"
        
        if not checkpoint_file.exists():
            return {"status": "not_found"}
        
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            return {
                "status": "found",
                "current_step": checkpoint.get("current_step", "unknown"),
                "processing_log": checkpoint.get("processing_log", []),
                "session_id": session_id
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all processing sessions"""
        log_dir = Path(self.config.global_config.log_dir)
        sessions = []
        
        if log_dir.exists():
            for session_dir in log_dir.iterdir():
                if session_dir.is_dir():
                    checkpoint_file = session_dir / f"{session_dir.name}_checkpoint.json"
                    if checkpoint_file.exists():
                        try:
                            with open(checkpoint_file, 'r') as f:
                                checkpoint = json.load(f)
                            
                            sessions.append({
                                "session_id": session_dir.name,
                                "current_step": checkpoint.get("current_step", "unknown"),
                                "pdf_name": checkpoint.get("pdf_metadata", {}).get("pdf_name", "unknown"),
                                "last_update": checkpoint_file.stat().st_mtime
                            })
                        except Exception:
                            continue
        
        return sorted(sessions, key=lambda x: x["last_update"], reverse=True)
