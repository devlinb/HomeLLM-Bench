"""System metrics collection for comprehensive benchmarking"""

import psutil
import platform
import socket
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False


class SystemMetricsCollector:
    """Collects detailed system information and metrics"""
    
    def __init__(self):
        self.gpu_available = PYNVML_AVAILABLE
        if self.gpu_available:
            try:
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except:
                self.gpu_available = False
                self.gpu_count = 0
        else:
            self.gpu_count = 0
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get detailed CPU information"""
        cpu_info = {
            'brand': platform.processor() or 'Unknown',
            'architecture': platform.machine(),
            'cores_physical': psutil.cpu_count(logical=False),
            'cores_logical': psutil.cpu_count(logical=True),
            'frequency_max_mhz': None,
            'frequency_current_mhz': None,
            'usage_percent': psutil.cpu_percent(interval=1),
            'usage_per_core': psutil.cpu_percent(percpu=True, interval=1)
        }
        
        # Get CPU frequency if available
        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_info['frequency_max_mhz'] = freq.max
                cpu_info['frequency_current_mhz'] = freq.current
        except:
            pass
        
        return cpu_info
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'used_percent': memory.percent,
            'swap_total_gb': round(swap.total / (1024**3), 2),
            'swap_used_gb': round(swap.used / (1024**3), 2),
            'swap_percent': swap.percent
        }
    
    def get_disk_info(self) -> List[Dict[str, Any]]:
        """Get disk usage information"""
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'filesystem': partition.fstype,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'free_gb': round(usage.free / (1024**3), 2),
                    'used_percent': round((usage.used / usage.total) * 100, 1)
                })
            except (PermissionError, OSError):
                # Skip inaccessible partitions
                continue
        
        return disks
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Get GPU information and current metrics"""
        if not self.gpu_available:
            return []
        
        gpus = []
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    memory_util = util.memory
                except:
                    gpu_util = None
                    memory_util = None
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                except:
                    power = None
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    graphics_clock = None
                    memory_clock = None
                
                gpus.append({
                    'index': i,
                    'name': name,
                    'memory_total_gb': round(memory_info.total / (1024**3), 2),
                    'memory_used_gb': round(memory_info.used / (1024**3), 2),
                    'memory_free_gb': round(memory_info.free / (1024**3), 2),
                    'memory_used_percent': round((memory_info.used / memory_info.total) * 100, 1),
                    'utilization_gpu_percent': gpu_util,
                    'utilization_memory_percent': memory_util,
                    'temperature_celsius': temp,
                    'power_usage_watts': power,
                    'graphics_clock_mhz': graphics_clock,
                    'memory_clock_mhz': memory_clock
                })
                
            except Exception as e:
                # If we can't get info for this GPU, skip it
                continue
        
        return gpus
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get network interface information"""
        network_info = {
            'hostname': socket.gethostname(),
            'interfaces': []
        }
        
        # Get network interfaces
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        
        for interface_name, addresses in net_if_addrs.items():
            if interface_name in net_if_stats:
                stats = net_if_stats[interface_name]
                interface_info = {
                    'name': interface_name,
                    'is_up': stats.isup,
                    'speed_mbps': stats.speed,
                    'addresses': []
                }
                
                for addr in addresses:
                    interface_info['addresses'].append({
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask
                    })
                
                network_info['interfaces'].append(interface_info)
        
        return network_info
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get current process information"""
        process = psutil.Process()
        
        return {
            'pid': process.pid,
            'name': process.name(),
            'cpu_percent': process.cpu_percent(),
            'memory_mb': round(process.memory_info().rss / (1024**2), 2),
            'memory_percent': process.memory_percent(),
            'num_threads': process.num_threads(),
            'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
        }
    
    def get_complete_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'timestamp': datetime.now().isoformat(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            },
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disks': self.get_disk_info(),
            'gpus': self.get_gpu_info(),
            'network': self.get_network_info(),
            'process': self.get_process_info(),
            'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
        }
    
    def get_runtime_metrics(self) -> Dict[str, Any]:
        """Get current runtime metrics (lighter than complete system info)"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory': self.get_memory_info(),
            'gpus': self.get_gpu_info(),
            'process': self.get_process_info()
        }


if __name__ == "__main__":
    # Test the system metrics collector
    collector = SystemMetricsCollector()
    
    print("Testing System Metrics Collector")
    print("=" * 40)
    
    print("\nCPU Info:")
    cpu_info = collector.get_cpu_info()
    for key, value in cpu_info.items():
        print(f"  {key}: {value}")
    
    print("\nMemory Info:")
    memory_info = collector.get_memory_info()
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
    
    print("\nGPU Info:")
    gpu_info = collector.get_gpu_info()
    for i, gpu in enumerate(gpu_info):
        print(f"  GPU {i}:")
        for key, value in gpu.items():
            print(f"    {key}: {value}")
    
    print("\nComplete system info collected successfully!")