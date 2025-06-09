"""
Performance monitoring plugin for RopeJumpCounter

This plugin provides real-time performance monitoring capabilities including
FPS tracking, latency measurement, and resource usage statistics.
"""

import sys
import os
import time
import logging
from typing import Dict, Any
from dataclasses import dataclass

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.plugin_manager import BasePlugin
from src.core.event_bus import get_event_bus, EventType


class Plugin(BasePlugin):
    """Performance monitoring plugin"""
    
    def __init__(self):
        super().__init__(
            name="performance_monitor",
            version="1.0.0",
            description="Monitors application performance metrics",
            author="RopeJumpCounter Team"
        )
        self.monitoring_thread = None
        self.stop_monitoring = False
        self.interval = 1.0  # seconds
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        try:
            self.interval = config.get('interval', 1.0)
            self.config = config
            return True
        except Exception as e:
            print(f"Failed to initialize performance monitor plugin: {e}")
            return False
    
    def start(self) -> bool:
        """Start performance monitoring"""
        try:
            self.stop_monitoring = False
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            return True
        except Exception as e:
            print(f"Failed to start performance monitor plugin: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop performance monitoring"""
        try:
            self.stop_monitoring = True
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=2.0)
            return True
        except Exception as e:
            print(f"Failed to stop performance monitor plugin: {e}")
            return False
    
    def cleanup(self):
        """Cleanup plugin resources"""
        self.stop()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        event_bus = get_event_bus()
        
        while not self.stop_monitoring:
            try:
                # Collect performance metrics
                metrics = {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_used': psutil.virtual_memory().used / (1024**3),  # GB
                    'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
                    'timestamp': time.time()
                }
                
                # Publish metrics through event bus
                event_bus.publish(EventType.PERFORMANCE_UPDATE, metrics, "performance_monitor")
                
                # Wait for next interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
                time.sleep(self.interval) 