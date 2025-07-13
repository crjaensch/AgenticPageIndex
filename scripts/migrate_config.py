#!/usr/bin/env python3
"""
Migration script to convert legacy PageIndex config.yaml to new hierarchical structure
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description='Migrate legacy PageIndex configuration')
    parser.add_argument('legacy_config', help='Path to legacy config.yaml file')
    parser.add_argument('--output', '-o', help='Output path for new config (default: config_new.yaml)')
    parser.add_argument('--backup', '-b', action='store_true', help='Create backup of legacy config')
    
    args = parser.parse_args()
    
    legacy_path = Path(args.legacy_config)
    if not legacy_path.exists():
        print(f"Error: Legacy config file not found: {legacy_path}")
        sys.exit(1)
    
    # Create backup if requested
    if args.backup:
        backup_path = legacy_path.with_suffix('.yaml.backup')
        backup_path.write_text(legacy_path.read_text())
        print(f"Backup created: {backup_path}")
    
    # Migrate configuration
    config_manager = ConfigManager()
    output_path = args.output or str(legacy_path.with_name("config_new.yaml"))
    
    try:
        new_config_path = config_manager.migrate_legacy_config(str(legacy_path), output_path)
        print("Migration completed successfully!")
        print(f"New config saved to: {new_config_path}")
        print("\nNew configuration structure:")
        
        # Display new config
        with open(new_config_path, 'r') as f:
            print(f.read())
            
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
