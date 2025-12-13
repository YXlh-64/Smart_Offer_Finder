"""
Batch processing module for Smart Offer Finder.

This module provides functionality to process questions in batch mode
according to a standardized JSON format for automated evaluation.

Input Format:
{
  "equipe": "NomDeLEquipe",
  "question": {
    "ID categorie": {
      "ID question": "question"
    }
  }
}

Output Format:
{
  "equipe": "NomDeLEquipe",
  "reponses": {
    "Nom de l'offre": {
      "ID question": "Réponse"
    }
  }
}
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from datetime import datetime

from .chat import cached_chain_invoke
from . import chat as chat_module


class BatchProcessor:
    """Process multiple questions in batch mode."""
    
    def __init__(self):
        """Initialize the batch processor."""
        if chat_module.chain is None:
            raise RuntimeError(
                "Chain not initialized. Please ensure the chat module is properly set up."
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input JSON format.
        
        Args:
            input_data: Input dictionary to validate
            
        Raises:
            ValueError: If input format is invalid
        """
        if "equipe" not in input_data:
            raise ValueError("Missing required field 'equipe' in input")
        
        if "question" not in input_data:
            raise ValueError("Missing required field 'question' in input")
        
        if not isinstance(input_data["question"], dict):
            raise ValueError("Field 'question' must be a dictionary")
        
        # Check that each category contains questions
        for category_id, questions in input_data["question"].items():
            if not isinstance(questions, dict):
                raise ValueError(
                    f"Category '{category_id}' must contain a dictionary of questions"
                )
            
            for question_id, question_text in questions.items():
                if not isinstance(question_text, str):
                    raise ValueError(
                        f"Question ID '{question_id}' in category '{category_id}' "
                        f"must be a string"
                    )
    
    def extract_offer_name(self, answer: str, sources: List[str]) -> str:
        """
        Extract offer name from answer and sources.
        
        This function attempts to identify the main offer being discussed
        in the response. It looks at the sources and the answer content.
        
        Args:
            answer: Generated answer text
            sources: List of source documents used
            
        Returns:
            Offer name or "Offre_Generale" if not identified
        """
        # Try to extract from sources
        # Sources typically contain file paths like "data/Offres/Idoom_ADSL.pdf"
        offer_keywords = []
        
        for source in sources:
            # Extract filename from path
            filename = Path(source).stem
            
            # Common offer patterns
            if "offre" in filename.lower() or "offer" in filename.lower():
                offer_keywords.append(filename)
            elif any(keyword in filename.lower() for keyword in [
                "idoom", "fléxy", "flexy", "4g", "adsl", "fibre", 
                "internet", "mobile", "fixe"
            ]):
                offer_keywords.append(filename)
        
        if offer_keywords:
            # Return the first identified offer
            return offer_keywords[0]
        
        # Fallback: try to extract from answer
        # Look for common patterns in Arabic and French
        answer_lower = answer.lower()
        
        offer_patterns = [
            "idoom adsl", "idoom fibre", "idoom 4g lte",
            "fléxy", "flexy",
            "forfait", "offre",
        ]
        
        for pattern in offer_patterns:
            if pattern in answer_lower:
                return pattern.replace(" ", "_").title()
        
        # Default fallback
        return "Offre_Generale"
    
    def process_question(
        self, 
        question: str, 
        category_id: str, 
        question_id: str
    ) -> Dict[str, Any]:
        """
        Process a single question and return the result.
        
        Args:
            question: The question text
            category_id: Category identifier
            question_id: Question identifier
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        print(f"  Processing [{category_id}][{question_id}]: {question[:60]}...")
        
        start_time = time.time()
        
        try:
            # Use the cached chain invoke
            result = cached_chain_invoke(question)
            
            answer = result.get("answer", "Aucune réponse générée.")
            source_docs = result.get("source_documents", []) or []
            sources = [doc.metadata.get("source", "unknown") for doc in source_docs]
            is_cache_hit = result.get("cache_hit", False)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "answer": answer,
                "sources": sources,
                "category_id": category_id,
                "question_id": question_id,
                "processing_time_ms": processing_time,
                "cache_hit": is_cache_hit,
                "success": True
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "answer": f"Erreur lors du traitement: {str(e)}",
                "sources": [],
                "category_id": category_id,
                "question_id": question_id,
                "processing_time_ms": processing_time,
                "cache_hit": False,
                "success": False,
                "error": str(e)
            }
    
    def process_batch(
        self, 
        input_data: Dict[str, Any],
        group_by_offer: bool = True
    ) -> Dict[str, Any]:
        """
        Process all questions in batch mode.
        
        Args:
            input_data: Input dictionary following the required format
            group_by_offer: If True, group responses by detected offer name
            
        Returns:
            Dictionary following the output format
        """
        # Validate input
        self.validate_input(input_data)
        
        equipe = input_data["equipe"]
        questions_dict = input_data["question"]
        
        print(f"\n{'='*80}")
        print(f"🚀 Démarrage du traitement batch pour l'équipe: {equipe}")
        print(f"{'='*80}\n")
        
        # Process all questions
        all_results = []
        total_questions = sum(len(q) for q in questions_dict.values())
        processed_count = 0
        
        batch_start_time = time.time()
        
        for category_id, questions in questions_dict.items():
            print(f"\n📂 Catégorie: {category_id} ({len(questions)} questions)")
            
            for question_id, question_text in questions.items():
                processed_count += 1
                print(f"  [{processed_count}/{total_questions}] ", end="")
                
                result = self.process_question(
                    question_text, 
                    category_id, 
                    question_id
                )
                all_results.append(result)
                
                status = "✅ Cache" if result.get("cache_hit") else "✅ Nouveau"
                if not result.get("success"):
                    status = "❌ Erreur"
                
                print(f"    {status} ({result['processing_time_ms']:.0f}ms)")
        
        batch_total_time = (time.time() - batch_start_time) * 1000
        
        # Build output structure
        if group_by_offer:
            # Group responses by detected offer
            output = {
                "equipe": equipe,
                "reponses": {}
            }
            
            for result in all_results:
                # Extract offer name from the result
                offer_name = self.extract_offer_name(
                    result["answer"], 
                    result["sources"]
                )
                
                # Initialize offer dict if not exists
                if offer_name not in output["reponses"]:
                    output["reponses"][offer_name] = {}
                
                # Add the answer
                output["reponses"][offer_name][result["question_id"]] = result["answer"]
        else:
            # Single "All" category
            output = {
                "equipe": equipe,
                "reponses": {
                    "Toutes_Offres": {}
                }
            }
            
            for result in all_results:
                output["reponses"]["Toutes_Offres"][result["question_id"]] = result["answer"]
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"✅ Traitement terminé!")
        print(f"{'='*80}")
        print(f"  Questions traitées: {processed_count}/{total_questions}")
        print(f"  Temps total: {batch_total_time:.2f}ms ({batch_total_time/1000:.2f}s)")
        print(f"  Temps moyen par question: {batch_total_time/total_questions:.2f}ms")
        
        cache_hits = sum(1 for r in all_results if r.get("cache_hit"))
        errors = sum(1 for r in all_results if not r.get("success"))
        
        print(f"  Cache hits: {cache_hits}/{total_questions} ({cache_hits/total_questions*100:.1f}%)")
        if errors > 0:
            print(f"  ⚠️  Erreurs: {errors}")
        print(f"{'='*80}\n")
        
        return output
    
    def save_output(self, output_data: Dict[str, Any], output_path: str) -> None:
        """
        Save output data to JSON file.
        
        Args:
            output_data: Output dictionary to save
            output_path: Path to output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Résultats sauvegardés: {output_path}")
    
    def load_input(self, input_path: str) -> Dict[str, Any]:
        """
        Load input data from JSON file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Input dictionary
        """
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def process_batch_from_file(
    input_file: str, 
    output_file: Optional[str] = None,
    group_by_offer: bool = True
) -> Dict[str, Any]:
    """
    Process batch questions from an input JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file (optional)
        group_by_offer: If True, group responses by detected offer name
        
    Returns:
        Output dictionary
    """
    processor = BatchProcessor()
    
    # Load input
    print(f"📥 Chargement de: {input_file}")
    input_data = processor.load_input(input_file)
    
    # Process batch
    output_data = processor.process_batch(input_data, group_by_offer=group_by_offer)
    
    # Save output if path provided
    if output_file:
        processor.save_output(output_data, output_file)
    
    return output_data


if __name__ == "__main__":
    # Allow running this module with: python -m src.batch_processor input.json [output.json]
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Process batch questions from a JSON file (module mode).",
        epilog="Run with: python -m src.batch_processor input.json [output.json]"
    )

    parser.add_argument("input_file", help="Path to input JSON file")
    parser.add_argument("output_file", nargs="?", help="Optional path to output JSON file")
    parser.add_argument("--no-group-by-offer", action="store_true", help="Do not group responses by offer")

    args = parser.parse_args()

    try:
        # Initialize the chain before processing
        if not chat_module.initialize_chain():
            print("[batch_processor] Failed to initialize chain. Aborting.", file=sys.stderr)
            sys.exit(1)

        group_by_offer = not args.no_group_by_offer
        output = process_batch_from_file(args.input_file, args.output_file, group_by_offer=group_by_offer)

        if not args.output_file:
            # Print to stdout
            print(json.dumps(output, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Error running batch processor: {e}", file=sys.stderr)
        sys.exit(1)
