#!/usr/bin/env python3
"""
Script to fix absolute imports to relative imports in the lexora package.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Get the directory depth to calculate relative imports
    rel_path = file_path.relative_to(Path('lexora'))
    depth = len(rel_path.parts) - 1  # -1 for the file itself
    
    # Patterns to replace
    patterns = [
        (r'^from exceptions import', f'from {"." * (depth + 1)}exceptions import'),
        (r'^from models\.', f'from {"." * (depth + 1)}models.'),
        (r'^from utils\.', f'from {"." * (depth + 1)}utils.'),
        (r'^from tools import', f'from {"." * (depth + 1)}tools import'),
        (r'^from tools\.', f'from {"." * (depth + 1)}tools.'),
        (r'^from llm\.', f'from {"." * (depth + 1)}llm.'),
        (r'^from vector_db\.', f'from {"." * (depth + 1)}vector_db.'),
        (r'^from rag_agent\.', f'from {"." * (depth + 1)}rag_agent.'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {file_path}")
        return True
    return False

def main():
    """Main function to fix all imports."""
    lexora_dir = Path('lexora')
    
    if not lexora_dir.exists():
        print("Error: lexora directory not found!")
        return
    
    fixed_count = 0
    for py_file in lexora_dir.rglob('*.py'):
        if py_file.name == '__init__.py' and py_file.parent == lexora_dir:
            # Skip the main __init__.py as we already fixed it
            continue
        
        if fix_imports_in_file(py_file):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
