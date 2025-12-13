#!/usr/bin/env python3
"""
Script to merge data2.json (NGBSS guides) into data.json (conventions)
"""

import json
from pathlib import Path

def merge_json_files():
    """Merge data2.json into data.json"""
    
    # Define file paths
    data_json_path = Path("data/raw/data.json")
    data2_json_path = Path("data/raw/data2.json")
    backup_path = Path("data/raw/data.json.backup")
    
    print("🔄 Starting merge process...")
    
    # Read data.json (conventions)
    print(f"📖 Reading {data_json_path}...")
    with open(data_json_path, 'r', encoding='utf-8') as f:
        conventions_data = json.load(f)
    print(f"✅ Loaded {len(conventions_data)} convention entries")
    
    # Read data2.json (NGBSS guides)
    print(f"📖 Reading {data2_json_path}...")
    with open(data2_json_path, 'r', encoding='utf-8') as f:
        ngbss_data = json.load(f)
    print(f"✅ Loaded {len(ngbss_data)} NGBSS guide entries")
    
    # Create backup of original data.json
    print(f"💾 Creating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(conventions_data, f, ensure_ascii=False, indent=2)
    print("✅ Backup created")
    
    # Transform NGBSS entries to have metadata for consistency
    print("🔧 Transforming NGBSS entries...")
    transformed_ngbss = []
    for idx, entry in enumerate(ngbss_data, 1):
        # Convert steps array to formatted text
        steps_text = "\n\n".join([
            f"**{step_key}**: {step_value}"
            for step_dict in entry.get("Steps", [])
            for step_key, step_value in step_dict.items()
        ])
        
        transformed_entry = {
            "metadata": {
                "source_filename": "Guide NGBSS",
                "source_folder": "Guide NGBSS",
                "document_title": entry.get("Process_Title", f"Guide NGBSS Process {idx}"),
                "language": "fr",
                "created_date": "2025",
                "file_path": f"./data/raw/data2.json#{idx}",
                "document_type": "ngbss_guide",
                "process_title": entry.get("Process_Title", "")
            },
            "content": f"**{entry.get('Process_Title', 'Guide NGBSS')}**\n\n{steps_text}",
            "steps": entry.get("Steps", [])  # Keep original steps structure
        }
        transformed_ngbss.append(transformed_entry)
    print(f"✅ Transformed {len(transformed_ngbss)} NGBSS entries")
    
    # Merge: append NGBSS data to conventions data
    print("🔗 Merging data...")
    merged_data = conventions_data + transformed_ngbss
    print(f"✅ Total entries after merge: {len(merged_data)}")
    
    # Write merged data back to data.json
    print(f"💾 Writing merged data to {data_json_path}...")
    with open(data_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    print("✅ Merged data written successfully")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MERGE SUMMARY")
    print("="*60)
    print(f"Convention entries: {len(conventions_data)}")
    print(f"NGBSS guide entries: {len(ngbss_data)}")
    print(f"Total merged entries: {len(merged_data)}")
    print(f"Backup saved at: {backup_path}")
    print(f"Merged file: {data_json_path}")
    print("="*60)
    print("\n✅ Merge completed successfully!")
    
    # Show sample of NGBSS entries added
    print("\n📋 Sample of added NGBSS entries:")
    for i, entry in enumerate(transformed_ngbss[:3], 1):
        print(f"\n{i}. {entry['metadata']['document_title']}")
        print(f"   Steps count: {len(entry.get('steps', []))}")

if __name__ == "__main__":
    try:
        merge_json_files()
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        raise
