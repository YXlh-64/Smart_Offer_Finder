"""
Timing Visualization Module

Provides visual representations of timing data for different phases:
- Retrieval (Vectorstore search)
- Reranking (Optional BGE reranker)
- Generation (LLM generation)

Includes:
- Console bar charts
- ASCII visualizations
- JSON export for dashboards
- Real-time tracking
"""

import json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import math


@dataclass
class TimingData:
    """Data class for timing information"""
    retrieval: float = 0.0
    reranking: float = 0.0
    generation: float = 0.0
    total: float = 0.0
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class TimingVisualizer:
    """Visualize timing data with various output formats"""
    
    def __init__(self):
        self.history: List[TimingData] = []
    
    def add_timing(self, retrieval: float, reranking: float, generation: float):
        """Add timing data to history"""
        total = retrieval + reranking + generation
        timing = TimingData(
            retrieval=retrieval,
            reranking=reranking,
            generation=generation,
            total=total,
            timestamp=datetime.now().isoformat()
        )
        self.history.append(timing)
        return timing
    
    def get_average_timings(self) -> Dict[str, float]:
        """Get average timings across all recorded data"""
        if not self.history:
            return {"retrieval": 0, "reranking": 0, "generation": 0, "total": 0}
        
        avg_retrieval = sum(t.retrieval for t in self.history) / len(self.history)
        avg_reranking = sum(t.reranking for t in self.history) / len(self.history)
        avg_generation = sum(t.generation for t in self.history) / len(self.history)
        avg_total = sum(t.total for t in self.history) / len(self.history)
        
        return {
            "retrieval": avg_retrieval,
            "reranking": avg_reranking,
            "generation": avg_generation,
            "total": avg_total
        }
    
    @staticmethod
    def create_bar_chart(retrieval: float, reranking: float, generation: float) -> str:
        """
        Create a horizontal bar chart for timing breakdown
        
        Args:
            retrieval: Retrieval time in ms
            reranking: Reranking time in ms
            generation: Generation time in ms
            
        Returns:
            String representation of the chart
        """
        total = retrieval + reranking + generation
        
        if total == 0:
            return "No timing data"
        
        # Calculate percentages and bar lengths
        retrieval_pct = (retrieval / total) * 100
        reranking_pct = (reranking / total) * 100
        generation_pct = (generation / total) * 100
        
        bar_width = 40
        retrieval_bar_len = int((retrieval_pct / 100) * bar_width)
        reranking_bar_len = int((reranking_pct / 100) * bar_width)
        generation_bar_len = int((generation_pct / 100) * bar_width)
        
        # Create bars
        chart = "\n⏱️  TIMING BREAKDOWN (Per Phase)\n"
        chart += "=" * 75 + "\n"
        
        # Retrieval
        bar = "█" * retrieval_bar_len + "░" * (bar_width - retrieval_bar_len)
        chart += f"🔍 Retrieval     [{bar}] {retrieval_pct:>5.1f}%  ({retrieval:>7.2f}ms)\n"
        
        # Reranking
        if reranking > 0:
            bar = "█" * reranking_bar_len + "░" * (bar_width - reranking_bar_len)
            chart += f"🎯 Reranking     [{bar}] {reranking_pct:>5.1f}%  ({reranking:>7.2f}ms)\n"
        
        # Generation
        bar = "█" * generation_bar_len + "░" * (bar_width - generation_bar_len)
        chart += f"✨ Generation    [{bar}] {generation_pct:>5.1f}%  ({generation:>7.2f}ms)\n"
        
        chart += "-" * 75 + "\n"
        chart += f"{'Total Time':<16} {'':>40} ({total:>7.2f}ms)\n"
        chart += "=" * 75 + "\n"
        
        return chart
    
    @staticmethod
    def create_pie_chart(retrieval: float, reranking: float, generation: float) -> str:
        """
        Create ASCII pie chart for timing breakdown
        
        Args:
            retrieval: Retrieval time in ms
            reranking: Reranking time in ms
            generation: Generation time in ms
            
        Returns:
            String representation of pie chart
        """
        total = retrieval + reranking + generation
        
        if total == 0:
            return "No timing data"
        
        # Calculate percentages
        retrieval_pct = (retrieval / total) * 100
        reranking_pct = (reranking / total) * 100
        generation_pct = (generation / total) * 100
        
        chart = "\n📊 TIMING PIE CHART\n"
        chart += "=" * 60 + "\n"
        
        # Simple text-based pie representation
        chart += "\nPhase Distribution:\n"
        chart += f"  🔍 Retrieval:   {retrieval_pct:>5.1f}% ({retrieval:>7.2f}ms)\n"
        if reranking > 0:
            chart += f"  🎯 Reranking:   {reranking_pct:>5.1f}% ({reranking:>7.2f}ms)\n"
        chart += f"  ✨ Generation:  {generation_pct:>5.1f}% ({generation:>7.2f}ms)\n"
        chart += "-" * 60 + "\n"
        chart += f"  Total:          100.0% ({total:>7.2f}ms)\n"
        chart += "=" * 60 + "\n"
        
        return chart
    
    @staticmethod
    def create_comparison_table(timings_list: List[TimingData]) -> str:
        """
        Create a comparison table of multiple timing runs
        
        Args:
            timings_list: List of TimingData objects
            
        Returns:
            String representation of comparison table
        """
        if not timings_list:
            return "No timing data available"
        
        table = "\n📈 TIMING COMPARISON (Multiple Runs)\n"
        table += "=" * 90 + "\n"
        table += f"{'Run':<5} {'Retrieval':<15} {'Reranking':<15} {'Generation':<15} {'Total':<15}\n"
        table += "-" * 90 + "\n"
        
        for idx, timing in enumerate(timings_list, 1):
            table += f"{idx:<5} {timing.retrieval:>8.2f}ms      {timing.reranking:>8.2f}ms      {timing.generation:>8.2f}ms      {timing.total:>8.2f}ms\n"
        
        # Add statistics
        table += "-" * 90 + "\n"
        
        avg_retrieval = sum(t.retrieval for t in timings_list) / len(timings_list)
        avg_reranking = sum(t.reranking for t in timings_list) / len(timings_list)
        avg_generation = sum(t.generation for t in timings_list) / len(timings_list)
        avg_total = sum(t.total for t in timings_list) / len(timings_list)
        
        table += f"{'AVG':<5} {avg_retrieval:>8.2f}ms      {avg_reranking:>8.2f}ms      {avg_generation:>8.2f}ms      {avg_total:>8.2f}ms\n"
        table += "=" * 90 + "\n"
        
        return table
    
    @staticmethod
    def create_timeline_chart(retrieval: float, reranking: float, generation: float) -> str:
        """
        Create a timeline showing when each phase occurs
        
        Args:
            retrieval: Retrieval time in ms
            reranking: Reranking time in ms
            generation: Generation time in ms
            
        Returns:
            String representation of timeline
        """
        total = retrieval + reranking + generation
        
        if total == 0:
            return "No timing data"
        
        # Calculate positions
        scale = 70  # Total width of timeline
        
        retrieval_width = int((retrieval / total) * scale)
        reranking_width = int((reranking / total) * scale)
        generation_width = int((generation / total) * scale)
        
        chart = "\n⏲️  EXECUTION TIMELINE\n"
        chart += "=" * 80 + "\n"
        chart += "Time progression (left to right):\n\n"
        
        # Phase 1: Retrieval
        chart += "┌" + "─" * retrieval_width + "┐\n"
        chart += "│ 🔍 RETRIEVAL" + " " * (retrieval_width - 12) + "│\n"
        chart += f"│ {retrieval:.2f}ms" + " " * (retrieval_width - len(f"{retrieval:.2f}ms")) + "│\n"
        chart += "└" + "─" * retrieval_width + "┘\n"
        
        if reranking > 0:
            # Phase 2: Reranking
            chart += "        ┌" + "─" * reranking_width + "┐\n"
            chart += "        │ 🎯 RERANKING" + " " * (reranking_width - 12) + "│\n"
            chart += f"        │ {reranking:.2f}ms" + " " * (reranking_width - len(f"{reranking:.2f}ms")) + "│\n"
            chart += "        └" + "─" * reranking_width + "┘\n"
        
        # Phase 3: Generation
        generation_start = retrieval_width + (reranking_width if reranking > 0 else 0)
        generation_offset = " " * (generation_start + 8)
        chart += f"{generation_offset}┌" + "─" * generation_width + "┐\n"
        chart += f"{generation_offset}│ ✨ GENERATION" + " " * (generation_width - 14) + "│\n"
        chart += f"{generation_offset}│ {generation:.2f}ms" + " " * (generation_width - len(f"{generation:.2f}ms")) + "│\n"
        chart += f"{generation_offset}└" + "─" * generation_width + "┘\n"
        
        chart += "\n" + "─" * 80 + "\n"
        chart += f"Total Duration: {total:.2f}ms\n"
        chart += "=" * 80 + "\n"
        
        return chart
    
    def export_json(self, filepath: str):
        """Export timing history as JSON"""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_records": len(self.history),
            "timings": [t.to_dict() for t in self.history],
            "average": self.get_average_timings()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_all_visualizations(self, retrieval: float, reranking: float, generation: float):
        """Print all available visualizations"""
        print(self.create_bar_chart(retrieval, reranking, generation))
        print(self.create_timeline_chart(retrieval, reranking, generation))
        if self.history:
            print(self.create_comparison_table(self.history))


# Global visualizer instance
_visualizer = TimingVisualizer()


def visualize_timing(retrieval: float, reranking: float, generation: float):
    """
    Main function to visualize timing data
    
    Args:
        retrieval: Retrieval time in milliseconds
        reranking: Reranking time in milliseconds
        generation: Generation time in milliseconds
    """
    global _visualizer
    
    # Add to history
    _visualizer.add_timing(retrieval, reranking, generation)
    
    # Print visualizations
    _visualizer.print_all_visualizations(retrieval, reranking, generation)


def get_visualizer() -> TimingVisualizer:
    """Get the global visualizer instance"""
    return _visualizer
