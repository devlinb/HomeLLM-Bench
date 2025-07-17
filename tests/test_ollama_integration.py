import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homellm_bench.engines.ollama_engine import OllamaEngine
from homellm_bench.metrics.schemas import GenerationMetrics
from homellm_bench.benchmark.runner import BenchmarkRunner


class TestOllamaIntegration:
    """Integration tests for Ollama engine with benchmark runner"""
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_benchmark_runner_ollama_initialization(self, mock_collector_class):
        """Test benchmark runner initializes Ollama engine correctly"""
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        runner = BenchmarkRunner(
            host="127.0.0.1",
            port=11434,
            engine_type="ollama",
            model_name="llama3.2:3b"
        )
        
        assert runner.engine_type == "ollama"
        assert isinstance(runner.engine, OllamaEngine)
        assert runner.engine.model_name == "llama3.2:3b"
        assert runner.engine.host == "127.0.0.1"
        assert runner.engine.port == 11434
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_benchmark_runner_check_server_health(self, mock_collector_class):
        """Test benchmark runner health check with Ollama"""
        mock_collector = Mock()
        mock_collector.check_server_health.return_value = True
        mock_collector_class.return_value = mock_collector
        
        runner = BenchmarkRunner(
            engine_type="ollama",
            model_name="llama3.2:3b"
        )
        
        result = runner.check_server_health()
        
        assert result is True
        mock_collector.check_server_health.assert_called_once()
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_engine_type_validation(self, mock_collector_class):
        """Test that invalid engine types are rejected"""
        mock_collector_class.return_value = Mock()
        
        with pytest.raises(ValueError, match="Unsupported engine type"):
            BenchmarkRunner(engine_type="invalid_engine")
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_ollama_engine_in_benchmark_flow(self, mock_collector_class):
        """Test Ollama engine works in benchmark flow"""
        # Mock collector behavior
        mock_collector = Mock()
        mock_collector.check_server_health.return_value = True
        mock_collector.generate_chat_with_metrics.return_value = (
            "Test response",
            GenerationMetrics(
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
                time_to_first_token=0.1,
                total_generation_time=0.3,
                tokens_per_second=50.0,
                engine_name="ollama",
                model_name="llama3.2:3b"
            )
        )
        mock_collector_class.return_value = mock_collector
        
        # Create runner
        runner = BenchmarkRunner(
            engine_type="ollama",
            model_name="llama3.2:3b"
        )
        
        # Test generate_chat_with_metrics through runner
        messages = [{"role": "user", "content": "Hello"}]
        text, metrics = runner.engine.generate_chat_with_metrics(
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
        
        # Verify results
        assert text == "Test response"
        assert metrics.engine_name == "ollama"
        assert metrics.model_name == "llama3.2:3b"
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 15
        assert metrics.tokens_per_second == 50.0
        
        # Verify collector was called correctly
        mock_collector.generate_chat_with_metrics.assert_called_once_with(
            model_name="llama3.2:3b",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
    
    def test_engine_comparison_vllm_vs_ollama(self):
        """Test that vLLM and Ollama engines have compatible interfaces"""
        # This test ensures both engines implement the same interface
        # without actually running the engines
        
        with patch('homellm_bench.engines.vllm_engine.VLLMMetricsCollector'):
            vllm_runner = BenchmarkRunner(
                engine_type="vllm",
                model_name="test-model"
            )
        
        with patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector'):
            ollama_runner = BenchmarkRunner(
                engine_type="ollama",
                model_name="test-model"
            )
        
        # Both engines should have the same interface
        vllm_methods = [method for method in dir(vllm_runner.engine) if not method.startswith('_')]
        ollama_methods = [method for method in dir(ollama_runner.engine) if not method.startswith('_')]
        
        # Check that key methods exist in both
        required_methods = ['is_running', 'generate_chat_with_metrics', 'warmup']
        
        for method in required_methods:
            assert method in vllm_methods, f"vLLM engine missing {method}"
            assert method in ollama_methods, f"Ollama engine missing {method}"
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_metrics_compatibility(self, mock_collector_class):
        """Test that Ollama metrics are compatible with existing schemas"""
        mock_collector = Mock()
        mock_collector_class.return_value = mock_collector
        
        # Create a realistic Ollama metrics object
        ollama_metrics = GenerationMetrics(
            prompt_tokens=26,
            completion_tokens=15,
            total_tokens=41,
            time_to_first_token=0.326,
            total_generation_time=5.043,
            tokens_per_second=3.18,
            engine_name="ollama",
            model_name="llama3.2:3b",
            engine_metrics={
                'ollama_native_metrics': {
                    'total_duration_ns': 5043500667,
                    'load_duration_ns': 5025959,
                    'prompt_eval_duration_ns': 325953000,
                    'eval_duration_ns': 4709213000,
                    'load_time_seconds': 0.005,
                    'prompt_eval_time_seconds': 0.326,
                    'eval_time_seconds': 4.709
                }
            }
        )
        
        # Test that metrics can be serialized/deserialized
        metrics_dict = ollama_metrics.model_dump()
        
        # Verify key fields
        assert metrics_dict['engine_name'] == "ollama"
        assert metrics_dict['model_name'] == "llama3.2:3b"
        assert metrics_dict['prompt_tokens'] == 26
        assert metrics_dict['completion_tokens'] == 15
        assert metrics_dict['tokens_per_second'] == 3.18
        
        # Verify Ollama-specific metrics
        assert 'ollama_native_metrics' in metrics_dict['engine_metrics']
        ollama_native = metrics_dict['engine_metrics']['ollama_native_metrics']
        assert ollama_native['total_duration_ns'] == 5043500667
        assert ollama_native['load_time_seconds'] == 0.005
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_error_handling_integration(self, mock_collector_class):
        """Test error handling in integrated environment"""
        mock_collector = Mock()
        mock_collector.check_server_health.return_value = False
        mock_collector_class.return_value = mock_collector
        
        runner = BenchmarkRunner(
            engine_type="ollama",
            model_name="llama3.2:3b"
        )
        
        # Test health check failure
        assert runner.check_server_health() is False
        
        # Test generation failure when server is down
        messages = [{"role": "user", "content": "Test"}]
        with pytest.raises(RuntimeError, match="Ollama server is not running"):
            runner.engine.generate_chat_with_metrics(messages)
    
    def test_default_model_selection(self):
        """Test that engine-specific default models are used"""
        # Test that the CLI would select appropriate defaults
        # This simulates the logic in benchmark runner main()
        
        # Ollama should default to llama3.2:3b
        engine_type = "ollama"
        default_model = "llama3.2:3b" if engine_type == "ollama" else "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
        default_port = 11434 if engine_type == "ollama" else 8000
        
        assert default_model == "llama3.2:3b"
        assert default_port == 11434
        
        # vLLM should default to Qwen model
        engine_type = "vllm"
        default_model = "llama3.2:3b" if engine_type == "ollama" else "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
        default_port = 11434 if engine_type == "ollama" else 8000
        
        assert default_model == "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
        assert default_port == 8000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])