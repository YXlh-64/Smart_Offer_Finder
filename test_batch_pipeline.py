#!/usr/bin/env python3
"""
Test script for the batch processing pipeline.
This script validates the implementation without requiring actual document ingestion.
"""

import json
import sys
from pathlib import Path

def validate_input_format(data):
    """Validate input JSON format."""
    errors = []
    
    if "equipe" not in data:
        errors.append("Missing 'equipe' field")
    elif not isinstance(data["equipe"], str):
        errors.append("'equipe' must be a string")
    
    if "question" not in data:
        errors.append("Missing 'question' field")
    elif not isinstance(data["question"], dict):
        errors.append("'question' must be a dictionary")
    else:
        for cat_id, questions in data["question"].items():
            if not isinstance(questions, dict):
                errors.append(f"Category '{cat_id}' must contain a dictionary")
            else:
                for q_id, q_text in questions.items():
                    if not isinstance(q_text, str):
                        errors.append(f"Question '{q_id}' in category '{cat_id}' must be a string")
    
    return errors

def validate_output_format(data):
    """Validate output JSON format."""
    errors = []
    
    if "equipe" not in data:
        errors.append("Missing 'equipe' field")
    elif not isinstance(data["equipe"], str):
        errors.append("'equipe' must be a string")
    
    if "reponses" not in data:
        errors.append("Missing 'reponses' field")
    elif not isinstance(data["reponses"], dict):
        errors.append("'reponses' must be a dictionary")
    else:
        for offer_name, answers in data["reponses"].items():
            if not isinstance(answers, dict):
                errors.append(f"Offer '{offer_name}' must contain a dictionary")
            else:
                for q_id, answer_text in answers.items():
                    if not isinstance(answer_text, str):
                        errors.append(f"Answer for question '{q_id}' must be a string")
    
    return errors

def test_example_files():
    """Test example files for format compliance."""
    
    print("="*80)
    print("🧪 Testing Batch Processing Pipeline")
    print("="*80)
    print()
    
    examples_dir = Path(__file__).parent / "examples"
    
    # Test input files
    input_files = [
        "batch_input_example.json",
        "batch_input_bilingual.json"
    ]
    
    print("📥 Testing Input Files:")
    print("-" * 80)
    
    for filename in input_files:
        filepath = examples_dir / filename
        
        if not filepath.exists():
            print(f"❌ {filename}: File not found")
            continue
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            errors = validate_input_format(data)
            
            if errors:
                print(f"❌ {filename}:")
                for error in errors:
                    print(f"   - {error}")
            else:
                # Count questions
                total_questions = sum(len(q) for q in data["question"].values())
                categories = len(data["question"])
                
                print(f"✅ {filename}:")
                print(f"   - Team: {data['equipe']}")
                print(f"   - Categories: {categories}")
                print(f"   - Questions: {total_questions}")
        
        except json.JSONDecodeError as e:
            print(f"❌ {filename}: Invalid JSON - {e}")
        except Exception as e:
            print(f"❌ {filename}: Error - {e}")
    
    print()
    
    # Test output file
    output_file = examples_dir / "batch_output_example.json"
    
    print("📤 Testing Output File:")
    print("-" * 80)
    
    if not output_file.exists():
        print(f"❌ batch_output_example.json: File not found")
    else:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            errors = validate_output_format(data)
            
            if errors:
                print(f"❌ batch_output_example.json:")
                for error in errors:
                    print(f"   - {error}")
            else:
                # Count responses
                total_responses = sum(len(r) for r in data["reponses"].values())
                offers = len(data["reponses"])
                
                print(f"✅ batch_output_example.json:")
                print(f"   - Team: {data['equipe']}")
                print(f"   - Offers: {offers}")
                print(f"   - Responses: {total_responses}")
                print(f"   - Offer names: {', '.join(data['reponses'].keys())}")
        
        except json.JSONDecodeError as e:
            print(f"❌ batch_output_example.json: Invalid JSON - {e}")
        except Exception as e:
            print(f"❌ batch_output_example.json: Error - {e}")
    
    print()
    print("="*80)
    print("✅ Format Validation Complete")
    print("="*80)
    print()
    print("📚 Next Steps:")
    print("   1. Ensure documents are ingested: python -m src.ingest")
    print("   2. Start Ollama: ollama serve")
    print("   3. Test batch processing: python batch_process.py examples/batch_input_example.json test_output.json")
    print("   4. Or start API server: python main.py")
    print()

def test_module_imports():
    """Test that required modules can be imported."""
    
    print("🔧 Testing Module Imports:")
    print("-" * 80)
    
    modules_to_test = [
        ("src.batch_processor", "BatchProcessor"),
        ("src.chat", "initialize_chain"),
        ("src.config", "get_settings"),
    ]
    
    all_ok = True
    
    for module_name, item_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[item_name])
            item = getattr(module, item_name)
            print(f"✅ {module_name}.{item_name}")
        except ImportError as e:
            print(f"❌ {module_name}.{item_name}: Import error - {e}")
            all_ok = False
        except AttributeError as e:
            print(f"❌ {module_name}.{item_name}: Not found - {e}")
            all_ok = False
        except Exception as e:
            print(f"⚠️  {module_name}.{item_name}: Warning - {e}")
    
    print()
    
    if all_ok:
        print("✅ All modules imported successfully")
    else:
        print("⚠️  Some modules failed to import (this is expected if dependencies are not installed)")
    
    print()

if __name__ == "__main__":
    print()
    test_module_imports()
    test_example_files()
