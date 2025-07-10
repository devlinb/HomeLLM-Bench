"""
Simple exception handling for HomeLLM-Bench with actionable error messages
"""
import sys
from typing import Optional


class BenchmarkError(Exception):
    """Base benchmark error with actionable messages"""
    
    def __init__(self, message: str, suggestion: str = None, exit_code: int = 1):
        self.message = message
        self.suggestion = suggestion
        self.exit_code = exit_code
        super().__init__(message)
    
    def print_and_exit(self):
        """Print error message with suggestion and exit"""
        print(f"❌ Error: {self.message}")
        if self.suggestion:
            print(f"💡 Solution: {self.suggestion}")
        sys.exit(self.exit_code)


def handle_server_connection_error(host: str, port: int):
    """Handle vLLM server connection issues"""
    BenchmarkError(
        f"Cannot connect to vLLM server at {host}:{port}",
        f"Start vLLM server first: python start_vllm.py --model your_model --port {port}"
    ).print_and_exit()


def handle_model_config_error(model_name: str):
    """Handle model configuration issues"""
    BenchmarkError(
        f"Model '{model_name}' not found or invalid configuration",
        "Check model name or add to model registry. Use a model from the registry or provide full path"
    ).print_and_exit()


def handle_model_file_not_found(model_path: str):
    """Handle missing model files"""
    BenchmarkError(
        f"Model file not found: {model_path}",
        "Check the model path exists. For HuggingFace models, ensure they're downloaded"
    ).print_and_exit()


def handle_gpu_memory_error():
    """Handle GPU out of memory errors"""
    BenchmarkError(
        "GPU out of memory during model loading or inference",
        "Reduce context size with --context-size or use smaller model. Check GPU memory with nvidia-smi"
    ).print_and_exit()


def handle_system_resource_error(resource: str):
    """Handle system resource issues"""
    BenchmarkError(
        f"System resource issue: {resource}",
        "Check system resources (RAM, GPU memory) and close other applications"
    ).print_and_exit()


def handle_missing_dependency(dependency: str):
    """Handle missing Python dependencies"""
    BenchmarkError(
        f"Missing required dependency: {dependency}",
        f"Install dependency: pip install {dependency}"
    ).print_and_exit()


def handle_conversation_data_error(data_path: str):
    """Handle conversation data issues"""
    BenchmarkError(
        f"Cannot load conversation data from {data_path}",
        "Check that conversation data files exist and are properly formatted JSON"
    ).print_and_exit()


def handle_file_permission_error(file_path: str):
    """Handle file permission issues"""
    BenchmarkError(
        f"Permission denied accessing file: {file_path}",
        "Check file permissions and ensure you have read/write access to the directory"
    ).print_and_exit()


def handle_port_in_use_error(port: int):
    """Handle port already in use"""
    BenchmarkError(
        f"Port {port} is already in use",
        f"Use a different port: --port {port + 1} or stop the process using port {port}"
    ).print_and_exit()


def handle_invalid_argument_error(argument: str, value: str, expected: str):
    """Handle invalid CLI arguments"""
    BenchmarkError(
        f"Invalid argument {argument}={value}",
        f"Expected: {expected}"
    ).print_and_exit()


def handle_network_error(host: str, port: int):
    """Handle general network errors"""
    BenchmarkError(
        f"Network error connecting to {host}:{port}",
        "Check network connectivity and firewall settings"
    ).print_and_exit()


def handle_timeout_error(operation: str, timeout: int):
    """Handle operation timeouts"""
    BenchmarkError(
        f"Operation '{operation}' timed out after {timeout} seconds",
        "Server may be overloaded or generation taking too long. Try reducing context size or max tokens"
    ).print_and_exit()


def handle_keyboard_interrupt():
    """Handle user cancellation gracefully"""
    print("\n🛑 Benchmark interrupted by user")
    print("💡 Cleaning up and shutting down...")
    sys.exit(0)


def handle_unexpected_error(error: Exception, context: str):
    """Handle unexpected errors with context"""
    BenchmarkError(
        f"Unexpected error during {context}: {str(error)[:100]}",
        "This is an unexpected error. Please report this issue with the full error details"
    ).print_and_exit()


def safe_execute(func, context: str, *args, **kwargs):
    """
    Execute function with comprehensive error handling
    
    Args:
        func: Function to execute
        context: Description of operation for error messages
        *args, **kwargs: Arguments to pass to function
    
    Returns:
        Function result or exits on error
    """
    try:
        return func(*args, **kwargs)
    
    except KeyboardInterrupt:
        handle_keyboard_interrupt()
    
    except (ConnectionError, ConnectionRefusedError) as e:
        if "127.0.0.1" in str(e) or "localhost" in str(e):
            handle_server_connection_error("127.0.0.1", 8001)
        else:
            handle_network_error("unknown", 0)
    
    except FileNotFoundError as e:
        file_path = str(e).split("'")[1] if "'" in str(e) else str(e)
        if "model" in file_path.lower():
            handle_model_file_not_found(file_path)
        elif "conversation" in file_path.lower() or "data" in file_path.lower():
            handle_conversation_data_error(file_path)
        else:
            handle_file_permission_error(file_path)
    
    except PermissionError as e:
        file_path = str(e).split("'")[1] if "'" in str(e) else "unknown file"
        handle_file_permission_error(file_path)
    
    except ImportError as e:
        dependency = str(e).split("'")[1] if "'" in str(e) else str(e)
        handle_missing_dependency(dependency)
    
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            handle_gpu_memory_error()
        else:
            handle_system_resource_error(str(e)[:50])
    
    except OSError as e:
        if "address already in use" in str(e).lower():
            handle_port_in_use_error(8001)  # Default port
        else:
            handle_network_error("unknown", 0)
    
    except (ValueError, TypeError) as e:
        if "port" in str(e).lower():
            handle_invalid_argument_error("port", "invalid", "1024-65535")
        elif "context" in str(e).lower():
            handle_invalid_argument_error("context-size", "invalid", "positive integer")
        else:
            handle_unexpected_error(e, context)
    
    except Exception as e:
        handle_unexpected_error(e, context)


# Import torch conditionally to avoid import errors
try:
    import torch
except ImportError:
    # Create a dummy torch module with OutOfMemoryError
    class _DummyTorch:
        class cuda:
            class OutOfMemoryError(Exception):
                pass
    torch = _DummyTorch()