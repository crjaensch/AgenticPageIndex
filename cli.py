#!/usr/bin/env python3
"""
Command-line interface for PageIndex Agent
"""

import argparse
import json
import sys
from pathlib import Path
from agent.pageindex_agent import PageIndexAgent

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='PageIndex Agent - Extract PDF document structure')
    
    # Main command arguments
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--output', '-o', help='Output file path (default: auto-generated)')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    
    # Configuration overrides
    parser.add_argument('--model', help='LLM model to use')
    parser.add_argument('--log-dir', help='Directory for processing logs')
    parser.add_argument('--add-summaries', action='store_true', help='Generate node summaries')
    parser.add_argument('--add-text', action='store_true', help='Include node text in output')
    parser.add_argument('--no-node-ids', action='store_true', help='Skip adding node IDs')
    parser.add_argument('--accuracy-threshold', type=float, help='Verification accuracy threshold')
    
    # Utility commands
    parser.add_argument('--list-sessions', action='store_true', help='List processing sessions')
    parser.add_argument('--session-status', help='Get status of specific session')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_sessions:
        agent = PageIndexAgent()
        sessions = agent.list_sessions()
        
        if not sessions:
            print("No processing sessions found")
            return
        
        print(f"Found {len(sessions)} processing sessions:")
        for session in sessions:
            print(f"  {session['session_id']}: {session['pdf_name']} - {session['current_step']}")
        return
    
    if args.session_status:
        agent = PageIndexAgent()
        status = agent.get_processing_status(args.session_status)
        
        if status['status'] == 'not_found':
            print(f"Session not found: {args.session_status}")
            sys.exit(1)
        elif status['status'] == 'error':
            print(f"Error reading session: {status['error']}")
            sys.exit(1)
        else:
            print(f"Session: {status['session_id']}")
            print(f"Current step: {status['current_step']}")
            print("Processing log:")
            for log_entry in status['processing_log']:
                print(f"  - {log_entry}")
        return
    
    # Validate PDF file
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Build configuration overrides
    config_overrides = {}
    
    if args.model:
        config_overrides.setdefault('global', {})['model'] = args.model
    
    if args.log_dir:
        config_overrides.setdefault('global', {})['log_dir'] = args.log_dir
    
    if args.accuracy_threshold:
        config_overrides.setdefault('structure_verifier', {})['accuracy_threshold'] = args.accuracy_threshold
    
    # Enhancement settings
    if not args.no_node_ids:
        config_overrides.setdefault('structure_processor', {})['if_add_node_id'] = 'yes'
    
    if args.add_summaries:
        config_overrides.setdefault('structure_processor', {})['if_add_node_summary'] = 'yes'
    
    if args.add_text:
        config_overrides.setdefault('structure_processor', {})['if_add_node_text'] = 'yes'
    
    try:
        # Initialize agent
        if args.verbose:
            print("Initializing PageIndex Agent...")
        
        agent = PageIndexAgent(config_overrides=config_overrides, verbose=args.verbose)
        
        # Process PDF
        if args.verbose:
            print(f"Processing PDF: {args.pdf_path}")
        
        result = agent.process_pdf(args.pdf_path)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            pdf_name = Path(args.pdf_path).stem
            output_path = Path(f"{pdf_name}_structure.json")
        
        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print("Processing completed successfully!")
        print(f"Results saved to: {output_path}")
        
        if args.verbose:
            print(f"\nDocument: {result['doc_name']}")
            if 'doc_description' in result:
                print(f"Description: {result['doc_description']}")
            
            # Count nodes
            def count_nodes(structure):
                if isinstance(structure, list):
                    count = len(structure)
                    for node in structure:
                        if 'nodes' in node:
                            count += count_nodes(node['nodes'])
                    return count
                return 1
            
            total_nodes = count_nodes(result['structure'])
            print(f"Total sections extracted: {total_nodes}")
        
    except Exception as e:
        print(f"Error: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
