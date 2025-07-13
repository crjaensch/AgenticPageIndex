# PageIndex Agent

An intelligent agent for extracting hierarchical structure from PDF documents using Large Language Models (LLMs) and the OpenAI Agent SDK.

## Features

- **Intelligent TOC Detection**: Automatically detects table of contents and analyzes structure
- **Multiple Extraction Strategies**: Supports documents with/without TOC, with dynamic fallback
- **Structure Verification**: Validates extracted structure and corrects errors automatically
- **Flexible Configuration**: Hierarchical configuration system with tool-specific settings
- **Diagnostic Logging**: Comprehensive logging for troubleshooting and state recovery
- **Enhancement Options**: Optional node IDs, summaries, and document descriptions
- **Agent-Based Architecture**: Uses OpenAI Agent SDK for intelligent orchestration

## Installation

```bash
pip install pageindex_agent
```

Or install from source:

```bash
git clone <repository_url>
cd pageindex_agent
pip install -e .
```

## Quick Start

### 1. Set up your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Process a PDF document

```python
from pageindex_agent.agent.pageindex_agent import PageIndexAgent

# Initialize agent
agent = PageIndexAgent()

# Process PDF
result = agent.process_pdf("document.pdf")

# Save results
import json
with open("structure.json", "w") as f:
    json.dump(result, f, indent=2)
```

### 3. Command-line usage

```bash
# Basic usage
pageindex document.pdf

# With options
pageindex document.pdf --add-summaries --model gpt-4o --output custom_output.json

# List processing sessions
pageindex --list-sessions

# Check session status
pageindex --session-status <session_id>
```

## Configuration

The agent uses a hierarchical configuration system. Create a `config.yaml` file:

```yaml
global:
  model: "gpt-4o-mini"
  log_dir: "./logs"

toc_detector:
  toc_check_page_num: 20

structure_processor:
  if_add_node_id: "yes"
  if_add_node_summary: "yes"
  if_add_doc_description: "yes"
```

### Migration from Legacy Config

If you have an old PageIndex configuration:

```bash
pageindex-migrate old_config.yaml --output new_config.yaml
```

## Architecture

The agent uses a pipeline of specialized tools:

1. **PDF Parser**: Extracts text and metadata from PDF
2. **TOC Detector**: Finds and analyzes table of contents
3. **Structure Extractor**: Extracts hierarchy using best strategy
4. **Structure Verifier**: Validates and corrects extracted structure
5. **Structure Processor**: Generates final tree with enhancements

### Processing Strategies

- **TOC with Pages**: Uses TOC that contains page numbers
- **TOC without Pages**: Uses TOC structure, matches content to pages
- **No TOC**: Analyzes document content to infer structure

The agent automatically selects the best strategy and implements fallbacks.

## Error Handling & Diagnostics

The agent provides comprehensive error handling:

- **Automatic Recovery**: Implements fallback strategies for failed tools
- **Diagnostic Logging**: Saves complete processing state for troubleshooting
- **Session Management**: Track and resume processing sessions
- **Recovery Suggestions**: Provides actionable suggestions for failures

## Examples

### Basic Document Processing

```python
from pageindex_agent.agent.pageindex_agent import PageIndexAgent

agent = PageIndexAgent()
result = agent.process_pdf("research_paper.pdf")

print(f"Document: {result['doc_name']}")
print(f"Description: {result.get('doc_description', 'N/A')}")

# Print structure
for section in result['structure']:
    print(f"- {section['title']} (pages {section['start_index']}-{section['end_index']})")
```

### Custom Configuration

```python
config_overrides = {
    "global": {"model": "gpt-4o"},
    "structure_processor": {
        "if_add_node_summary": "yes",
        "if_add_node_text": "yes"
    },
    "structure_verifier": {
        "accuracy_threshold": 0.8
    }
}

agent = PageIndexAgent(config_overrides=config_overrides)
result = agent.process_pdf("document.pdf")
```

### Batch Processing

```python
from pathlib import Path

agent = PageIndexAgent()
pdf_dir = Path("documents/")

for pdf_file in pdf_dir.glob("*.pdf"):
    try:
        result = agent.process_pdf(str(pdf_file))
        
        output_file = f"results/{pdf_file.stem}_structure.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
            
        print(f"✓ Processed: {pdf_file.name}")
    except Exception as e:
        print(f"✗ Failed: {pdf_file.name} - {e}")
```

## Testing

Run the test suite:

```bash
# Unit tests
python -m pytest pageindex_agent/tests/test_*.py -v

# Integration tests
python -m pytest pageindex_agent/tests/test_integration.py -v

# All tests
python -m pytest pageindex_agent/tests/ -v
```

## Output Format

The agent returns a structured JSON object:

```json
{
  "doc_name": "document.pdf",
  "doc_description": "A research paper about machine learning methods",
  "structure": [
    {
      "node_id": "0001",
      "title": "Introduction",
      "start_index": 1,
      "end_index": 5,
      "summary": "Overview of the research problem and objectives",
      "nodes": [
        {
          "node_id": "0002",
          "title": "Background",
          "start_index": 2,
          "end_index": 3
        }
      ]
    }
  ]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

- Issues: Report bugs and feature requests via GitHub Issues
- Documentation: See the `docs/` directory for detailed documentation
- Examples: Check the `examples/` directory for more usage patterns