#!/usr/bin/env python3
"""
Batch processing script for Smart Offer Finder.

This script allows you to process multiple questions in batch mode
using the standardized JSON format for evaluation pipelines.

Usage:
    python batch_process.py input.json output.json
    python batch_process.py input.json output.json --no-group-by-offer
    python batch_process.py input.json  # Output to stdout
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.batch_processor import BatchProcessor
from src.chat import initialize_chain


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Process batch questions for Smart Offer Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process and save to file
  python batch_process.py input.json output.json
  
  # Process without grouping by offer
  python batch_process.py input.json output.json --no-group-by-offer
  
  # Process and print to stdout
  python batch_process.py input.json
  
  # Show detailed help
  python batch_process.py --help

Input Format:
  {
    "equipe": "Team_Name",
    "question": {
      "category_01": {
        "1": "Question 1?",
        "2": "Question 2?"
      }
    }
  }

Output Format:
  {
    "equipe": "Team_Name",
    "reponses": {
      "Offer_Name": {
        "1": "Answer 1",
        "2": "Answer 2"
      }
    }
  }
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to input JSON file with questions"
    )
    
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Path to output JSON file for results (optional, prints to stdout if not provided)"
    )
    
    parser.add_argument(
        "--no-group-by-offer",
        action="store_true",
        help="Do not group responses by detected offer name (use single 'Toutes_Offres' category)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input_file}", file=sys.stderr)
        return 1
    
    try:
        # Initialize the chain
        print("🔧 Initializing Smart Offer Finder...", file=sys.stderr)
        initialize_chain()
        
        # Create processor
        processor = BatchProcessor()
        
        # Load input
        print(f"📥 Loading input from: {args.input_file}", file=sys.stderr)
        input_data = processor.load_input(args.input_file)
        
        # Process batch
        group_by_offer = not args.no_group_by_offer
        output_data = processor.process_batch(input_data, group_by_offer=group_by_offer)
        
        # Save or print output
        if args.output_file:
            processor.save_output(output_data, args.output_file)
            print(f"\n✅ Résultats sauvegardés dans: {args.output_file}", file=sys.stderr)
        else:
            # Print to stdout
            import json
            print("\n" + "="*80, file=sys.stderr)
            print("📄 Résultats (JSON):", file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            print(json.dumps(output_data, ensure_ascii=False, indent=2))
        
        return 0
        
    except ValueError as e:
        print(f"\n❌ Erreur de format: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n❌ Erreur: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
