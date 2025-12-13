"""
Script to fix file_path values in data.json by scanning actual file structure.
"""

import json
import os
from pathlib import Path

DATA_JSON_PATH = Path("data/raw/data.json")
DATA_FOLDER = Path("data")

def find_file_in_directory(filename: str, search_dir: Path) -> str | None:
    """
    Recursively search for a file by name in a directory.
    Returns the relative path from project root if found, None otherwise.
    """
    for root, dirs, files in os.walk(search_dir):
        # Skip the 'raw' folder to avoid matching json files
        if 'raw' in root:
            continue
        if 'chroma_db' in root:
            continue
            
        for file in files:
            if file == filename:
                full_path = Path(root) / file
                # Return path relative to project root with forward slashes
                return str(full_path).replace("\\", "/")
    return None


def fix_file_paths():
    """
    Load data.json, fix all file_path values, and save back.
    """
    print(f"📂 Loading {DATA_JSON_PATH}...")
    
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📝 Processing {len(data)} entries...")
    
    fixed_count = 0
    not_found_count = 0
    already_correct = 0
    
    for i, entry in enumerate(data):
        metadata = entry.get("metadata", {})
        current_path = metadata.get("file_path", "")
        
        if not current_path:
            continue
        
        # Extract just the filename from the current path
        filename = Path(current_path).name
        
        # Search for the actual file location
        actual_path = find_file_in_directory(filename, DATA_FOLDER)
        
        if actual_path:
            # Normalize current path for comparison
            normalized_current = current_path.replace("\\", "/").lstrip("./")
            
            if normalized_current != actual_path:
                print(f"  ✓ Fixed: {filename}")
                print(f"    Old: {current_path}")
                print(f"    New: {actual_path}")
                metadata["file_path"] = actual_path
                fixed_count += 1
            else:
                already_correct += 1
        else:
            print(f"  ⚠️ Not found: {filename}")
            not_found_count += 1
    
    # Save the updated data
    print(f"\n💾 Saving updated data.json...")
    with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Done!")
    print(f"   - Fixed: {fixed_count}")
    print(f"   - Already correct: {already_correct}")
    print(f"   - Not found: {not_found_count}")
    
    if fixed_count > 0:
        print(f"\n⚠️  Don't forget to re-run: python -m src.ingest")


if __name__ == "__main__":
    fix_file_paths()
