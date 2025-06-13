"""
Plugin Management System

Provides a plugin architecture for extending application functionality
with modular components. Supports dynamic loading, configuration,
and lifecycle management of plugins.
"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

from .exceptions import AppError

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin information"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    enabled: bool = True


class BasePlugin(ABC):
    """Base class for all plugins"""

    def __init__(self, name: str, version: str, description: str = "", author: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.enabled = False
        self.config = {}

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin
        
        Args:
            config: Plugin configuration
            
        Returns:
            bool: True if initialization successful
        """
        pass

    @abstractmethod
    def start(self) -> bool:
        """Start the plugin
        
        Returns:
            bool: True if start successful
        """
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Stop the plugin
        
        Returns:
            bool: True if stop successful
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Cleanup plugin resources"""
        pass

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name=self.name,
            version=self.version,
            description=self.description,
            author=self.author,
            dependencies=getattr(self, 'dependencies', []),
            enabled=self.enabled
        )


class PluginManager:
    """Manages plugin loading, configuration, and lifecycle"""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = Path(plugin_dir)
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.enabled_plugins: List[str] = []

        # Create plugin directory if it doesn't exist
        self.plugin_dir.mkdir(exist_ok=True)

    def discover_plugins(self) -> List[str]:
        """Discover available plugins in the plugin directory
        
        Returns:
            List of discovered plugin names
        """
        discovered = []

        if not self.plugin_dir.exists():
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist")
            return discovered

        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue

            plugin_name = plugin_file.stem
            try:
                # Try to import the plugin module
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                if spec and spec.loader:
                    discovered.append(plugin_name)
                    logger.debug(f"Discovered plugin: {plugin_name}")
            except Exception as e:
                logger.warning(f"Failed to discover plugin {plugin_name}: {e}")

        return discovered

    def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin by name
        
        Args:
            plugin_name: Name of the plugin to load
            
        Returns:
            bool: True if loading successful
        """
        try:
            plugin_file = self.plugin_dir / f"{plugin_name}.py"

            if not plugin_file.exists():
                raise AppError(f"Plugin file {plugin_file} not found")

            # Import the plugin module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the plugin class (should be named Plugin)
            if not hasattr(module, 'Plugin'):
                raise AppError(f"Plugin {plugin_name} does not have a Plugin class")

            plugin_class = module.Plugin

            if not issubclass(plugin_class, BasePlugin):
                raise AppError(f"Plugin {plugin_name} does not inherit from BasePlugin")

            # Create plugin instance
            plugin = plugin_class()

            # Load plugin configuration
            config = self._load_plugin_config(plugin_name)

            # Initialize plugin
            if plugin.initialize(config):
                self.plugins[plugin_name] = plugin
                self.plugin_configs[plugin_name] = config
                logger.info(f"Plugin {plugin_name} loaded successfully")
                return True
            else:
                logger.error(f"Failed to initialize plugin {plugin_name}")
                return False

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return False

    def load_all_plugins(self) -> List[str]:
        """Load all discovered plugins
        
        Returns:
            List of successfully loaded plugin names
        """
        discovered = self.discover_plugins()
        loaded = []

        for plugin_name in discovered:
            if self.load_plugin(plugin_name):
                loaded.append(plugin_name)

        logger.info(f"Loaded {len(loaded)} plugins: {loaded}")
        return loaded

    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a loaded plugin
        
        Args:
            plugin_name: Name of the plugin to enable
            
        Returns:
            bool: True if enabling successful
        """
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not loaded")
            return False

        plugin = self.plugins[plugin_name]

        try:
            if plugin.start():
                plugin.enabled = True
                self.enabled_plugins.append(plugin_name)
                logger.info(f"Plugin {plugin_name} enabled")
                return True
            else:
                logger.error(f"Failed to start plugin {plugin_name}")
                return False
        except Exception as e:
            logger.error(f"Error enabling plugin {plugin_name}: {e}")
            return False

    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable an enabled plugin
        
        Args:
            plugin_name: Name of the plugin to disable
            
        Returns:
            bool: True if disabling successful
        """
        if plugin_name not in self.enabled_plugins:
            logger.warning(f"Plugin {plugin_name} not enabled")
            return False

        plugin = self.plugins[plugin_name]

        try:
            if plugin.stop():
                plugin.enabled = False
                self.enabled_plugins.remove(plugin_name)
                logger.info(f"Plugin {plugin_name} disabled")
                return True
            else:
                logger.error(f"Failed to stop plugin {plugin_name}")
                return False
        except Exception as e:
            logger.error(f"Error disabling plugin {plugin_name}: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            bool: True if unloading successful
        """
        if plugin_name not in self.plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False

        # Disable first if enabled
        if plugin_name in self.enabled_plugins:
            self.disable_plugin(plugin_name)

        plugin = self.plugins[plugin_name]

        try:
            plugin.cleanup()
            del self.plugins[plugin_name]
            if plugin_name in self.plugin_configs:
                del self.plugin_configs[plugin_name]
            logger.info(f"Plugin {plugin_name} unloaded")
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a loaded plugin by name"""
        return self.plugins.get(plugin_name)

    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        plugin = self.get_plugin(plugin_name)
        return plugin.get_info() if plugin else None

    def list_plugins(self) -> List[PluginInfo]:
        """List all loaded plugins with their information"""
        return [plugin.get_info() for plugin in self.plugins.values()]

    def list_enabled_plugins(self) -> List[str]:
        """List names of enabled plugins"""
        return self.enabled_plugins.copy()

    def _load_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """Load plugin configuration from file"""
        config_file = self.plugin_dir / f"{plugin_name}.yaml"

        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config for plugin {plugin_name}: {e}")

        return {}

    def cleanup(self):
        """Cleanup all plugins"""
        for plugin_name in list(self.plugins.keys()):
            self.unload_plugin(plugin_name)
        logger.info("All plugins cleaned up")


# Global plugin manager instance
plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance"""
    return plugin_manager
