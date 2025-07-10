"""
Fast vLLM server detection using process list
"""
import psutil
import socket
import requests
from typing import Tuple, Optional
DEFAULT_HOST = "127.0.0.1"


def check_vllm_process(port: int) -> Tuple[bool, Optional[int]]:
    """
    Check if vLLM server process is running on specified port
    
    Returns:
        Tuple of (is_running, pid)
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                
                # Look for vLLM API server process
                if 'vllm.entrypoints.openai.api_server' in cmdline:
                    # Check if it's using our port
                    if f'--port {port}' in cmdline or f'--port={port}' in cmdline:
                        return True, proc.info['pid']
                    
                    # Also check for default port if no port specified
                    if port == 8001 and '--port' not in cmdline:
                        return True, proc.info['pid']
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
    except Exception:
        pass
    
    return False, None


def is_port_open(port: int, host: str = DEFAULT_HOST, timeout: float = 0.1) -> bool:
    """
    Quick check if port is open (with minimal timeout)
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False


def quick_health_check(port: int, host: str = DEFAULT_HOST, timeout: float = 0.5) -> bool:
    """
    Quick health check with minimal timeout
    """
    try:
        response = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def detect_vllm_server(port: int, host: str = DEFAULT_HOST) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Fast detection of vLLM server status
    
    Returns:
        Tuple of (status, pid, message)
        
    Status values:
        - "not_running": No vLLM server detected
        - "starting_up": vLLM process found but not listening yet
        - "ready": vLLM server ready and responding
        - "port_busy": Port in use by different service
    """
    
    # Step 1: Fast process check (no timeout)
    vllm_running, pid = check_vllm_process(port)
    
    if not vllm_running:
        # Check if port is busy with something else
        if is_port_open(port):
            return "port_busy", None, f"Port {port} is in use by a different service"
        return "not_running", None, f"No vLLM server detected on port {port}"
    
    # Step 2: Quick socket check
    if not is_port_open(port):
        return "starting_up", pid, f"vLLM server starting up (PID: {pid})"
    
    # Step 3: Optional quick health check
    if quick_health_check(port, host):
        return "ready", pid, f"vLLM server ready (PID: {pid})"
    
    # Port is open but not responding to health check
    return "starting_up", pid, f"vLLM server listening but not ready yet (PID: {pid})"


def get_vllm_process_info(pid: int) -> Optional[dict]:
    """Get detailed info about vLLM process"""
    try:
        proc = psutil.Process(pid)
        cmdline = ' '.join(proc.cmdline())
        
        # Extract model name from command line
        model_name = "unknown"
        if "--model" in cmdline:
            parts = cmdline.split("--model")
            if len(parts) > 1:
                model_part = parts[1].strip().split()[0]
                model_name = model_part
        
        return {
            "pid": pid,
            "model": model_name,
            "cmdline": cmdline,
            "memory_mb": proc.memory_info().rss / (1024 * 1024),
            "cpu_percent": proc.cpu_percent(),
            "create_time": proc.create_time()
        }
    except:
        return None


def wait_for_server_ready(port: int, host: str = DEFAULT_HOST, max_wait: int = 30) -> bool:
    """
    Wait for vLLM server to become ready with progress updates
    
    Returns:
        True if server becomes ready, False if timeout
    """
    import time
    
    print(f"Waiting for vLLM server to become ready...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        status, pid, message = detect_vllm_server(port, host)
        
        if status == "ready":
            print(f"✅ Server ready!")
            return True
        elif status == "not_running":
            print(f"❌ Server process stopped")
            return False
        elif status == "port_busy":
            print(f"❌ Port conflict detected")
            return False
        
        # Show progress
        elapsed = int(time.time() - start_time)
        print(f"⏳ {message} (waiting {elapsed}s/{max_wait}s)")
        time.sleep(2)
    
    print(f"❌ Server did not become ready within {max_wait} seconds")
    return False


if __name__ == "__main__":
    # Test the detection
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    
    print(f"Testing vLLM server detection on port {port}...")
    
    status, pid, message = detect_vllm_server(port)
    
    print(f"Status: {status}")
    print(f"PID: {pid}")
    print(f"Message: {message}")
    
    if pid:
        info = get_vllm_process_info(pid)
        if info:
            print(f"Model: {info['model']}")
            print(f"Memory: {info['memory_mb']:.1f}MB")
            print(f"CPU: {info['cpu_percent']:.1f}%")