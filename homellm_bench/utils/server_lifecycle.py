"""
Simple server process management for vLLM
"""
import subprocess
import signal
import os
import time
from typing import Optional


class SimpleServerProcess:
    """Simple vLLM server process with basic lifecycle management"""
    
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down server...")
            self.stop()
            exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start(self, cmd: list[str]) -> bool:
        """Start the server process"""
        try:
            self.process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid if os.name != 'nt' else None  # Process group for clean shutdown
            )
            
            print(f"Server started with PID {self.process.pid}")
            return True
            
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def wait(self):
        """Wait for server to finish"""
        if self.process:
            self.process.wait()
    
    def stop(self):
        """Stop the server process"""
        if not self.process:
            return
        
        try:
            # Send SIGTERM to process group for clean shutdown
            if os.name != 'nt':
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            
            # Give it a moment to shutdown gracefully
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()
            
            print("Server stopped")
            
        except Exception as e:
            print(f"Error stopping server: {e}")
        finally:
            self.process = None


# Global server instance
_server: Optional[SimpleServerProcess] = None


def get_server() -> SimpleServerProcess:
    """Get global server instance"""
    global _server
    if _server is None:
        _server = SimpleServerProcess()
    return _server


def cleanup_server():
    """Clean up server at exit"""
    global _server
    if _server:
        _server.stop()
        _server = None


# Register cleanup function
import atexit
atexit.register(cleanup_server)