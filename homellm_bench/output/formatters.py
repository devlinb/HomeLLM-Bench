import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from metrics.schemas import GenerationMetrics, ConversationBenchmarkResult


class BenchmarkFormatter:
    """Formats benchmark results for human and machine consumption"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_results(self, 
                    conversation_results: List["ConversationBenchmarkResult"],
                    system_info: Dict[str, Any],
                    config_info: Dict[str, Any],
                    run_timestamp: Optional[datetime] = None) -> Dict[str, str]:
        """Save results in multiple formats and return file paths"""
        
        if run_timestamp is None:
            run_timestamp = datetime.now()
        
        timestamp_str = run_timestamp.strftime("%Y%m%d_%H%M%S")
        base_filename = f"benchmark_{timestamp_str}"
        
        files_created = {}
        
        # Save CSV (machine parsable)
        csv_path = self.output_dir / f"{base_filename}.csv"
        self._save_csv(conversation_results, csv_path)
        files_created['csv'] = str(csv_path)
        
        # Save detailed JSON (machine parsable)
        json_path = self.output_dir / f"{base_filename}.json"
        self._save_json(conversation_results, system_info, config_info, run_timestamp, json_path)
        files_created['json'] = str(json_path)
        
        # Save Markdown report (human readable)
        md_path = self.output_dir / f"{base_filename}.md"
        self._save_markdown(conversation_results, system_info, config_info, run_timestamp, md_path)
        files_created['markdown'] = str(md_path)
        
        # Save system info
        sysinfo_path = self.output_dir / f"{base_filename}_system.json"
        self._save_system_info(system_info, sysinfo_path)
        files_created['system_info'] = str(sysinfo_path)
        
        return files_created
    
    def _save_csv(self, results: List["ConversationBenchmarkResult"], path: Path):
        """Save results as CSV for machine analysis"""
        with open(path, 'w', newline='') as csvfile:
            fieldnames = [
                'conversation_name',
                'total_turns',
                'total_time_seconds',
                'total_tokens_generated',
                'avg_tokens_per_second',
                'cache_effectiveness',
                'timestamp',
                'engine_name',
                'model_name',
                # Per-turn averages
                'avg_prompt_tokens',
                'avg_completion_tokens', 
                'avg_generation_time',
                'avg_time_to_first_token',
                # TTS metrics
                'tts_enabled',
                'total_audio_files',
                'avg_tts_rtf',
                'total_tts_time'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Calculate averages across turns
                turn_metrics = result.turn_metrics
                if turn_metrics:
                    avg_prompt_tokens = sum(m.prompt_tokens for m in turn_metrics) / len(turn_metrics)
                    avg_completion_tokens = sum(m.completion_tokens for m in turn_metrics) / len(turn_metrics)
                    avg_generation_time = sum(m.total_generation_time for m in turn_metrics) / len(turn_metrics)
                    avg_ttft = sum(m.time_to_first_token for m in turn_metrics) / len(turn_metrics)
                    
                    # Calculate TTS metrics
                    tts_turns = [m for m in turn_metrics if m.tts_metadata]
                    if tts_turns:
                        tts_enabled = True
                        total_audio_files = sum(m.tts_metadata.get("audio_files_created", 0) for m in tts_turns)
                        avg_tts_rtf = sum(m.tts_metadata.get("tts_rtf", 0) for m in tts_turns) / len(tts_turns)
                        total_tts_time = sum(m.tts_metadata.get("tts_processing_time", 0) for m in tts_turns)
                    else:
                        tts_enabled = False
                        total_audio_files = 0
                        avg_tts_rtf = 0
                        total_tts_time = 0
                else:
                    avg_prompt_tokens = avg_completion_tokens = avg_generation_time = avg_ttft = 0
                    tts_enabled = False
                    total_audio_files = 0
                    avg_tts_rtf = 0
                    total_tts_time = 0
                
                # Extract engine info from first turn if available
                engine_name = turn_metrics[0].engine_name if turn_metrics else "unknown"
                model_name = turn_metrics[0].model_name if turn_metrics else "unknown"
                
                writer.writerow({
                    'conversation_name': result.conversation_name,
                    'total_turns': result.total_turns,
                    'total_time_seconds': round(result.total_time, 3),
                    'total_tokens_generated': result.total_tokens_generated,
                    'avg_tokens_per_second': round(result.avg_tokens_per_second, 1),
                    'cache_effectiveness': round(result.cache_effectiveness or 0, 3),
                    'timestamp': result.timestamp.isoformat(),
                    'engine_name': engine_name,
                    'model_name': model_name,
                    'avg_prompt_tokens': round(avg_prompt_tokens, 1),
                    'avg_completion_tokens': round(avg_completion_tokens, 1),
                    'avg_generation_time': round(avg_generation_time, 3),
                    'avg_time_to_first_token': round(avg_ttft, 3),
                    'tts_enabled': tts_enabled,
                    'total_audio_files': total_audio_files,
                    'avg_tts_rtf': round(avg_tts_rtf, 3),
                    'total_tts_time': round(total_tts_time, 3)
                })
    
    def _save_json(self, results: List["ConversationBenchmarkResult"], 
                   system_info: Dict[str, Any], config_info: Dict[str, Any],
                   timestamp: datetime, path: Path):
        """Save detailed results as JSON"""
        data = {
            'benchmark_info': {
                'timestamp': timestamp.isoformat(),
                'version': '1.0',
                'tool': 'llm-benchmark'
            },
            'system_info': system_info,
            'config_info': config_info,
            'results': [result.model_dump() for result in results]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_markdown(self, results: List["ConversationBenchmarkResult"],
                      system_info: Dict[str, Any], config_info: Dict[str, Any], 
                      timestamp: datetime, path: Path):
        """Save human-readable markdown report"""
        
        with open(path, 'w') as f:
            f.write(f"# LLM Benchmark Report\n\n")
            f.write(f"**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # System Information
            f.write("## System Information\n\n")
            f.write("| Component | Details |\n")
            f.write("|-----------|----------|\n")
            
            if 'cpu' in system_info:
                cpu = system_info['cpu']
                f.write(f"| CPU | {cpu.get('brand', 'Unknown')} ({cpu.get('cores', 'N/A')} cores) |\n")
            
            if 'memory' in system_info:
                mem = system_info['memory']
                f.write(f"| RAM | {mem.get('total_gb', 'N/A')} GB |\n")
            
            if 'gpus' in system_info:
                for i, gpu in enumerate(system_info['gpus']):
                    f.write(f"| GPU {i} | {gpu.get('name', 'Unknown')} ({gpu.get('memory_gb', 'N/A')} GB) |\n")
            
            f.write("\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write("| Setting | Value |\n")
            f.write("|---------|-------|\n")
            for key, value in config_info.items():
                f.write(f"| {key} | {value} |\n")
            
            # Add engine info if available from results
            if results and results[0].turn_metrics:
                first_metric = results[0].turn_metrics[0]
                f.write(f"| Engine | {first_metric.engine_name} |\n")
                f.write(f"| Model | {first_metric.model_name} |\n")
            
            f.write("\n")
            
            # Overall Results Summary
            f.write("## Results Summary\n\n")
            if results:
                total_conversations = len(results)
                total_tokens = sum(r.total_tokens_generated for r in results)
                avg_tokens_per_sec = sum(r.avg_tokens_per_second for r in results) / len(results)
                total_time = sum(r.total_time for r in results)
                
                f.write(f"- **Total Conversations:** {total_conversations}\n")
                f.write(f"- **Total Tokens Generated:** {total_tokens:,}\n")
                f.write(f"- **Average Speed:** {avg_tokens_per_sec:.1f} tokens/second\n")
                f.write(f"- **Total Time:** {total_time:.1f} seconds\n\n")
            
            # Detailed Results
            f.write("## Detailed Results\n\n")
            f.write("| Conversation | Turns | Tokens | Speed (tok/s) | Time (s) | Cache Effectiveness |\n")
            f.write("|-------------|-------|--------|---------------|----------|--------------------|\n")
            
            for result in results:
                cache_str = f"{result.cache_effectiveness:.3f}s" if result.cache_effectiveness else "N/A"
                f.write(f"| {result.conversation_name} | {result.total_turns} | "
                       f"{result.total_tokens_generated} | {result.avg_tokens_per_second:.1f} | "
                       f"{result.total_time:.2f} | {cache_str} |\n")
            
            f.write("\n")
            
            # Per-Turn Metrics (for first conversation as example)
            if results and results[0].turn_metrics:
                f.write("## Sample Turn-by-Turn Metrics\n\n")
                f.write(f"*From conversation: {results[0].conversation_name}*\n\n")
                f.write("| Turn | Prompt Tokens | Completion Tokens | Speed (tok/s) | TTFT (ms) | TTS Files | TTS RTF |\n")
                f.write("|------|---------------|------------------|---------------|----------|-----------|----------|\n")
                
                for i, metric in enumerate(results[0].turn_metrics, 1):
                    ttft_ms = metric.time_to_first_token * 1000
                    
                    # Extract TTS metrics if available
                    tts_files = "N/A"
                    tts_rtf = "N/A"
                    if metric.tts_metadata:
                        tts_files = str(metric.tts_metadata.get("audio_files_created", "N/A"))
                        tts_rtf_val = metric.tts_metadata.get("tts_rtf", 0)
                        tts_rtf = f"{tts_rtf_val:.3f}" if tts_rtf_val else "N/A"
                    
                    f.write(f"| {i} | {metric.prompt_tokens} | {metric.completion_tokens} | "
                           f"{metric.tokens_per_second:.1f} | {ttft_ms:.0f} | {tts_files} | {tts_rtf} |\n")
                f.write("\n")
                
                # Add TTS Summary if any turns have TTS data
                tts_turns = [m for m in results[0].turn_metrics if m.tts_metadata]
                if tts_turns:
                    f.write("## TTS Performance Summary\n\n")
                    total_audio_files = sum(m.tts_metadata.get("audio_files_created", 0) for m in tts_turns)
                    total_tts_time = sum(m.tts_metadata.get("tts_processing_time", 0) for m in tts_turns)
                    avg_rtf = sum(m.tts_metadata.get("tts_rtf", 0) for m in tts_turns) / len(tts_turns)
                    
                    f.write(f"- **Total Audio Files Generated:** {total_audio_files}\n")
                    f.write(f"- **Total TTS Processing Time:** {total_tts_time:.2f} seconds\n")
                    f.write(f"- **Average TTS RTF:** {avg_rtf:.3f} (lower is better)\n")
                    f.write(f"- **TTS-Enabled Turns:** {len(tts_turns)}/{len(results[0].turn_metrics)}\n\n")
                    
                    # TTS performance interpretation
                    if avg_rtf < 0.5:
                        performance = "Excellent (2x+ faster than real-time)"
                    elif avg_rtf < 1.0:
                        performance = "Good (faster than real-time)"
                    elif avg_rtf < 2.0:
                        performance = "Fair (slower than real-time)"
                    else:
                        performance = "Poor (much slower than real-time)"
                    
                    f.write(f"**TTS Performance:** {performance}\n\n")
    
    def _save_system_info(self, system_info: Dict[str, Any], path: Path):
        """Save detailed system information"""
        with open(path, 'w') as f:
            json.dump(system_info, f, indent=2, default=str)


if __name__ == "__main__":
    # Test the formatter - import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from metrics.schemas import GenerationMetrics, ConversationBenchmarkResult
    
    # Create sample data
    sample_metrics = [
        GenerationMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            time_to_first_token=0.05,
            total_generation_time=0.4,
            tokens_per_second=50.0,
            engine_name="vllm",
            model_name="test-model"
        )
    ]
    
    sample_result = ConversationBenchmarkResult(
        conversation_name="Test Conversation",
        total_turns=1,
        total_time=0.4,
        turn_metrics=sample_metrics,
        avg_tokens_per_second=50.0,
        total_tokens_generated=20
    )
    
    formatter = BenchmarkFormatter()
    files = formatter.save_results(
        [sample_result],
        {"cpu": {"brand": "Test CPU", "cores": 8}},
        {"model": "test-model", "batch_size": 256}
    )
    
    print("Test files created:")
    for format_type, path in files.items():
        print(f"  {format_type}: {path}")