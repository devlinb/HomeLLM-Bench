"""
Simple port availability checking
"""
import socket
import sys
from ..config.constants import DEFAULT_HOST, PORT_CHECK_TIMEOUT


def check_port_available(port: int, host: str = DEFAULT_HOST) -> None:
    """Check if port is available. Exit with error if not."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(PORT_CHECK_TIMEOUT)
            result = sock.connect_ex((host, port))
            if result == 0:  # Port is in use
                print(f"Error: Port {port} is already in use")
                sys.exit(1)
    except Exception as e:
        print(f"Error checking port {port}: {e}")
        sys.exit(1)