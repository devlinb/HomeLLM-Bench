import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homellm_bench.metrics.ollama_collector import OllamaMetricsCollector
from homellm_bench.metrics.schemas import GenerationMetrics


class TestOllamaMetricsCollector:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.collector = OllamaMetricsCollector("127.0.0.1", 11434)
    
    def test_init(self):
        """Test collector initialization"""
        assert self.collector.server_host == "127.0.0.1"
        assert self.collector.server_port == 11434
        assert self.collector.base_url == "http://127.0.0.1:11434"
        assert self.collector.session is not None
    
    def test_nanoseconds_conversion(self):
        """Test nanoseconds to seconds conversion"""
        # Test conversion
        assert self.collector._convert_nanoseconds_to_seconds(1_000_000_000) == 1.0
        assert self.collector._convert_nanoseconds_to_seconds(500_000_000) == 0.5
        assert self.collector._convert_nanoseconds_to_seconds(0) == 0.0
    
    @patch('requests.Session.get')
    def test_check_server_health_success(self, mock_get):
        """Test successful health check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.collector.check_server_health()
        
        assert result is True
        mock_get.assert_called_once_with("http://127.0.0.1:11434/api/tags", timeout=5)
    
    @patch('requests.Session.get')
    def test_check_server_health_failure(self, mock_get):
        """Test failed health check"""
        mock_get.side_effect = Exception("Connection failed")
        
        result = self.collector.check_server_health()
        
        assert result is False
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_get_system_metrics(self, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock psutil responses
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(
            used=1024 * 1024 * 1024,  # 1GB in bytes
            percent=25.0
        )
        
        metrics = self.collector.get_system_metrics()
        
        assert metrics['cpu_percent'] == 45.5
        assert metrics['memory_used_mb'] == 1024.0
        assert metrics['memory_percent'] == 25.0
    
    @patch('requests.Session.post')
    def test_generate_chat_with_metrics_success(self, mock_post):
        """Test successful chat generation with metrics"""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "content": "Hello! How are you?"
            },
            "total_duration": 5043500667,  # ~5.04 seconds in nanoseconds
            "load_duration": 5025959,      # ~5ms
            "prompt_eval_duration": 325953000,  # ~326ms
            "eval_duration": 4709213000,   # ~4.7 seconds
            "prompt_eval_count": 26,
            "eval_count": 15
        }
        mock_post.return_value = mock_response
        
        # Mock system metrics
        with patch.object(self.collector, 'get_system_metrics') as mock_system:
            mock_system.return_value = {'cpu_percent': 50.0}
            
            messages = [{"role": "user", "content": "Hello"}]
            text, metrics = self.collector.generate_chat_with_metrics(
                model_name="llama3.2:3b",
                messages=messages,
                max_tokens=100
            )
        
        # Verify response
        assert text == "Hello! How are you?"
        assert isinstance(metrics, GenerationMetrics)
        
        # Verify metrics
        assert metrics.prompt_tokens == 26
        assert metrics.completion_tokens == 15
        assert metrics.total_tokens == 41
        assert metrics.engine_name == "ollama"
        assert metrics.model_name == "llama3.2:3b"
        
        # Verify timing calculations
        assert abs(metrics.total_generation_time - 5.0435) < 0.01  # ~5.04 seconds
        assert abs(metrics.time_to_first_token - 0.326) < 0.01     # ~326ms
        assert abs(metrics.tokens_per_second - 3.19) < 0.1        # 15 tokens / 4.7s ≈ 3.19
        
        # Verify engine metrics
        assert 'ollama_native_metrics' in metrics.engine_metrics
        ollama_metrics = metrics.engine_metrics['ollama_native_metrics']
        assert ollama_metrics['total_duration_ns'] == 5043500667
        assert ollama_metrics['load_duration_ns'] == 5025959
        assert ollama_metrics['prompt_eval_duration_ns'] == 325953000
        assert ollama_metrics['eval_duration_ns'] == 4709213000
    
    @patch('requests.Session.post')
    def test_generate_chat_with_metrics_failure(self, mock_post):
        """Test failed chat generation"""
        mock_post.side_effect = Exception("Connection failed")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(RuntimeError, match="Generation failed"):
            self.collector.generate_chat_with_metrics(
                model_name="llama3.2:3b",
                messages=messages
            )
    
    @patch('requests.Session.post')
    def test_generate_chat_with_metrics_zero_duration(self, mock_post):
        """Test chat generation with zero eval duration"""
        # Mock response with zero eval duration
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hi"},
            "total_duration": 1000000000,
            "load_duration": 0,
            "prompt_eval_duration": 1000000000,
            "eval_duration": 0,  # Zero duration
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        mock_post.return_value = mock_response
        
        with patch.object(self.collector, 'get_system_metrics'):
            messages = [{"role": "user", "content": "Hi"}]
            text, metrics = self.collector.generate_chat_with_metrics(
                model_name="test",
                messages=messages
            )
        
        # Should handle zero duration gracefully
        assert metrics.tokens_per_second == 0
        assert metrics.completion_tokens == 5
    
    @patch.object(OllamaMetricsCollector, 'generate_chat_with_metrics')
    def test_warmup_model_success(self, mock_generate):
        """Test successful model warmup"""
        mock_generate.return_value = ("Hi", Mock())
        
        result = self.collector.warmup_model("llama3.2:3b")
        
        assert result is True
        mock_generate.assert_called_once_with(
            model_name="llama3.2:3b",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0.0
        )
    
    @patch.object(OllamaMetricsCollector, 'generate_chat_with_metrics')
    def test_warmup_model_failure(self, mock_generate):
        """Test failed model warmup"""
        mock_generate.side_effect = Exception("Warmup failed")
        
        with patch('builtins.print'):  # Suppress print output
            result = self.collector.warmup_model("llama3.2:3b")
        
        assert result is False
    
    @patch('requests.Session.post')
    def test_request_data_format(self, mock_post):
        """Test that request data is formatted correctly for Ollama"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "test"},
            "total_duration": 1000000000,
            "load_duration": 0,
            "prompt_eval_duration": 500000000,
            "eval_duration": 500000000,
            "prompt_eval_count": 10,
            "eval_count": 5
        }
        mock_post.return_value = mock_response
        
        with patch.object(self.collector, 'get_system_metrics'):
            messages = [{"role": "user", "content": "Test"}]
            self.collector.generate_chat_with_metrics(
                model_name="test-model",
                messages=messages,
                max_tokens=50,
                temperature=0.5
            )
        
        # Verify request format
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        assert call_args[1]['json']['model'] == "test-model"
        assert call_args[1]['json']['messages'] == messages
        assert call_args[1]['json']['stream'] is False
        assert call_args[1]['json']['options']['temperature'] == 0.5
        assert call_args[1]['json']['options']['num_predict'] == 50
        assert call_args[0][0] == "http://127.0.0.1:11434/api/chat"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])