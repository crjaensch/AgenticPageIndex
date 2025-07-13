"""
Complete example showing how to use the PageIndex Agent
"""

import os
import json
from pathlib import Path
from pageindex_agent.agent.pageindex_agent import PageIndexAgent
from pageindex_agent.core.config import ConfigManager

def main():
    """Main example function"""
    
    # 1. Setup configuration (optional - uses defaults if not provided)
    config_overrides = {
        "global": {
            "model": "gpt-4o-mini",  # or "gpt-4o" for better quality
            "log_dir": "./processing_logs"
        },
        "toc_detector": {
            "toc_check_page_num": 20  # Check first 20 pages for TOC
        },
        "structure_processor": {
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",  # Generate summaries
            "if_add_doc_description": "yes",
            "if_add_node_text": "no"  # Don't include full text in output
        },
        "structure_verifier": {
            "accuracy_threshold": 0.7  # Accept 70% accuracy
        }
    }
    
    # 2. Initialize the agent
    # Make sure OPENAI_API_KEY is set in your environment
    agent = PageIndexAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        config_overrides=config_overrides
    )
    
    # 3. Process a PDF document
    pdf_path = "path/to/your/document.pdf"
    
    try:
        print(f"Processing PDF: {pdf_path}")
        result = agent.process_pdf(pdf_path)
        
        # 4. Save results
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{result['doc_name']}_structure.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("Processing completed successfully!")
        print(f"Results saved to: {output_file}")
        
        # 5. Display summary
        structure = result['structure']
        print(f"\nDocument: {result['doc_name']}")
        if 'doc_description' in result:
            print(f"Description: {result['doc_description']}")
        
        print("Structure extracted:")
        print_structure_summary(structure)
        
        return result
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        
        # Check for diagnostic information
        sessions = agent.list_sessions()
        if sessions:
            latest_session = sessions[0]
            print(f"Diagnostic information available for session: {latest_session['session_id']}")
            
            status = agent.get_processing_status(latest_session['session_id'])
            print(f"Last step completed: {status.get('current_step', 'unknown')}")
            
            log_dir = Path(agent.config["global"]["log_dir"]) / latest_session['session_id']
            print(f"Full diagnostic logs at: {log_dir}")
        
        raise


def print_structure_summary(structure, indent=0):
    """Print a summary of the extracted structure"""
    
    def print_node(node, level=0):
        prefix = "  " * level
        title = node.get('title', 'Untitled')
        start_page = node.get('start_index', '?')
        end_page = node.get('end_index', '?')
        
        print(f"{prefix}- {title} (pages {start_page}-{end_page})")
        
        # Print summary if available
        if 'summary' in node:
            summary = node['summary'][:100] + "..." if len(node['summary']) > 100 else node['summary']
            print(f"{prefix}  Summary: {summary}")
        
        # Print child nodes
        if 'nodes' in node:
            for child in node['nodes']:
                print_node(child, level + 1)
    
    if isinstance(structure, list):
        for node in structure:
            print_node(node)
    else:
        print_node(structure)


def migrate_legacy_config():
    """Example of migrating legacy configuration"""
    
    config_manager = ConfigManager()
    
    # Migrate old config.yaml to new format
    legacy_config_path = "old_config.yaml"
    if Path(legacy_config_path).exists():
        print("Migrating legacy configuration...")
        new_config_path = config_manager.migrate_legacy_config(
            legacy_config_path, 
            "config_new.yaml"
        )
        print(f"New configuration saved to: {new_config_path}")
    else:
        print("No legacy config found, using defaults")


def batch_process_pdfs(pdf_directory: str):
    """Example of batch processing multiple PDFs"""
    
    agent = PageIndexAgent()
    pdf_dir = Path(pdf_directory)
    results_dir = Path("batch_results")
    results_dir.mkdir(exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")
            result = agent.process_pdf(str(pdf_file))
            
            # Save individual result
            output_file = results_dir / f"{pdf_file.stem}_structure.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            successful += 1
            print(f"✓ Success: {pdf_file.name}")
            
        except Exception as e:
            failed += 1
            print(f"✗ Failed: {pdf_file.name} - {e}")
            
            # Save error log
            error_file = results_dir / f"{pdf_file.stem}_error.txt"
            with open(error_file, 'w') as f:
                f.write(f"Error processing {pdf_file.name}:\n{str(e)}")
    
    print("\nBatch processing completed:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_usage.py <pdf_path>                    # Process single PDF")
        print("  python example_usage.py --batch <directory>          # Process all PDFs in directory")
        print("  python example_usage.py --migrate                    # Migrate legacy config")
        sys.exit(1)
    
    if sys.argv[1] == "--migrate":
        migrate_legacy_config()
    elif sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Please provide directory path for batch processing")
            sys.exit(1)
        batch_process_pdfs(sys.argv[2])
    else:
        # Single PDF processing
        pdf_path = sys.argv[1]
        if not Path(pdf_path).exists():
            print(f"PDF file not found: {pdf_path}")
            sys.exit(1)
        
        main()
