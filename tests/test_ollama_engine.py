import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homellm_bench.engines.ollama_engine import OllamaEngine
from homellm_bench.metrics.schemas import GenerationMetrics


class TestOllamaEngine:
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = OllamaEngine(
            model_name="llama3.2:3b",
            host="127.0.0.1",
            port=11434
        )
    
    def test_init(self):
        """Test engine initialization"""
        assert self.engine.model_name == "llama3.2:3b"
        assert self.engine.host == "127.0.0.1"
        assert self.engine.port == 11434
        assert self.engine.metrics_collector is not None
    
    def test_init_with_defaults(self):
        """Test engine initialization with default values"""
        engine = OllamaEngine()
        
        assert engine.model_name == "llama3.2:3b"
        assert engine.host == "127.0.0.1"
        assert engine.port == 11434
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_is_running_success(self, mock_collector_class):
        """Test successful health check"""
        mock_collector = Mock()
        mock_collector.check_server_health.return_value = True
        mock_collector_class.return_value = mock_collector
        
        engine = OllamaEngine()
        result = engine.is_running()
        
        assert result is True
        mock_collector.check_server_health.assert_called_once()
    
    @patch('homellm_bench.engines.ollama_engine.OllamaMetricsCollector')
    def test_is_running_failure(self, mock_collector_class):
        """Test failed health check"""
        mock_collector = Mock()
        mock_collector.check_server_health.return_value = False
        mock_collector_class.return_value = mock_collector
        
        engine = OllamaEngine()
        result = engine.is_running()
        
        assert result is False
    
    @patch.object(OllamaEngine, 'is_running')
    def test_generate_chat_with_metrics_server_down(self, mock_is_running):
        """Test chat generation when server is down"""
        mock_is_running.return_value = False
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(RuntimeError, match="Ollama server is not running"):
            self.engine.generate_chat_with_metrics(messages)
    
    @patch.object(OllamaEngine, 'is_running')
    def test_generate_chat_with_metrics_success(self, mock_is_running):
        """Test successful chat generation"""
        mock_is_running.return_value = True
        
        # Mock the metrics collector
        mock_metrics = GenerationMetrics(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            time_to_first_token=0.1,
            total_generation_time=0.5,
            tokens_per_second=40.0,
            engine_name="ollama",
            model_name="llama3.2:3b"
        )
        
        with patch.object(self.engine.metrics_collector, 'generate_chat_with_metrics') as mock_generate:
            mock_generate.return_value = ("Hello there!", mock_metrics)
            
            messages = [{"role": "user", "content": "Hello"}]
            text, metrics = self.engine.generate_chat_with_metrics(
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )
        
        # Verify results
        assert text == "Hello there!"
        assert isinstance(metrics, GenerationMetrics)
        assert metrics.engine_name == "ollama"
        assert metrics.model_name == "llama3.2:3b"
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20
        
        # Verify correct parameters passed to collector
        mock_generate.assert_called_once_with(
            model_name="llama3.2:3b",
            messages=messages,
            max_tokens=100,
            temperature=0.7
        )
    
    @patch.object(OllamaEngine, 'is_running')
    def test_generate_chat_with_metrics_with_kwargs(self, mock_is_running):
        """Test chat generation with additional kwargs"""
        mock_is_running.return_value = True
        
        mock_metrics = GenerationMetrics(
            prompt_tokens=5,
            completion_tokens=15,
            total_tokens=20,
            time_to_first_token=0.05,
            total_generation_time=0.3,
            tokens_per_second=50.0,
            engine_name="ollama",
            model_name="llama3.2:3b"
        )
        
        with patch.object(self.engine.metrics_collector, 'generate_chat_with_metrics') as mock_generate:
            mock_generate.return_value = ("Response", mock_metrics)
            
            messages = [{"role": "user", "content": "Test"}]
            text, metrics = self.engine.generate_chat_with_metrics(
                messages=messages,
                max_tokens=50,
                temperature=0.2,
                top_p=0.9,
                custom_param="value"
            )
        
        # Verify kwargs are passed through
        mock_generate.assert_called_once_with(
            model_name="llama3.2:3b",
            messages=messages,
            max_tokens=50,
            temperature=0.2,
            top_p=0.9,
            custom_param="value"
        )
    
    @patch.object(OllamaEngine, 'is_running')
    def test_warmup_server_running(self, mock_is_running):
        """Test warmup when server is running"""
        mock_is_running.return_value = True
        
        with patch.object(self.engine.metrics_collector, 'warmup_model') as mock_warmup:
            mock_warmup.return_value = True
            
            result = self.engine.warmup()
        
        assert result is True
        mock_warmup.assert_called_once_with("llama3.2:3b")
    
    @patch.object(OllamaEngine, 'is_running')
    def test_warmup_server_not_running(self, mock_is_running):
        """Test warmup when server is not running"""
        mock_is_running.return_value = False
        
        result = self.engine.warmup()
        
        assert result is False
    
    @patch.object(OllamaEngine, 'is_running')
    def test_warmup_failure(self, mock_is_running):
        """Test warmup failure"""
        mock_is_running.return_value = True
        
        with patch.object(self.engine.metrics_collector, 'warmup_model') as mock_warmup:
            mock_warmup.return_value = False
            
            result = self.engine.warmup()
        
        assert result is False
    
    def test_engine_with_custom_model(self):
        """Test engine with custom model name"""
        engine = OllamaEngine(model_name="qwen2.5:3b")
        
        assert engine.model_name == "qwen2.5:3b"
        assert engine.host == "127.0.0.1"
        assert engine.port == 11434
    
    def test_engine_with_custom_host_port(self):
        """Test engine with custom host and port"""
        engine = OllamaEngine(
            model_name="phi3.5:3b",
            host="192.168.1.100",
            port=8080
        )
        
        assert engine.model_name == "phi3.5:3b"
        assert engine.host == "192.168.1.100"
        assert engine.port == 8080
        assert engine.metrics_collector.server_host == "192.168.1.100"
        assert engine.metrics_collector.server_port == 8080
    
    def test_metrics_collector_initialization(self):
        """Test that metrics collector is initialized with correct parameters"""
        engine = OllamaEngine(
            model_name="test-model",
            host="test-host",
            port=9999
        )
        
        assert engine.metrics_collector.server_host == "test-host"
        assert engine.metrics_collector.server_port == 9999
        assert engine.metrics_collector.base_url == "http://test-host:9999"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])