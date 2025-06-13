"""
Event Bus System

Provides a centralized event system for decoupled communication between
application components. Supports event publishing, subscription, and
asynchronous event handling.
"""

import asyncio
import threading
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Predefined event types"""
    JUMP_DETECTED = "jump_detected"
    FRAME_PROCESSED = "frame_processed"
    MODEL_LOADED = "model_loaded"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_UPDATE = "performance_update"
    CONFIG_CHANGED = "config_changed"
    APPLICATION_START = "application_start"
    APPLICATION_STOP = "application_stop"


@dataclass
class Event:
    """Event data structure"""
    type: EventType
    data: Any
    timestamp: datetime
    source: str

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """Centralized event bus for application communication"""

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._async_subscribers: Dict[EventType, List[Callable]] = {}
        self._event_history: List[Event] = []
        self._max_history: int = 1000
        self._lock = threading.Lock()
        self._running = False
        self._event_queue = asyncio.Queue()

    def subscribe(self, event_type: EventType, handler: Callable, async_handler: bool = False):
        """Subscribe to an event type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Function to call when event occurs
            async_handler: Whether the handler is async
        """
        with self._lock:
            if async_handler:
                if event_type not in self._async_subscribers:
                    self._async_subscribers[event_type] = []
                self._async_subscribers[event_type].append(handler)
            else:
                if event_type not in self._subscribers:
                    self._subscribers[event_type] = []
                self._subscribers[event_type].append(handler)

        logger.debug(f"Subscribed to {event_type.value} with {'async' if async_handler else 'sync'} handler")

    def unsubscribe(self, event_type: EventType, handler: Callable, async_handler: bool = False):
        """Unsubscribe from an event type"""
        with self._lock:
            if async_handler:
                if event_type in self._async_subscribers:
                    try:
                        self._async_subscribers[event_type].remove(handler)
                    except ValueError:
                        pass
            else:
                if event_type in self._subscribers:
                    try:
                        self._subscribers[event_type].remove(handler)
                    except ValueError:
                        pass

        logger.debug(f"Unsubscribed from {event_type.value}")

    def publish(self, event_type: EventType, data: Any = None, source: str = "unknown"):
        """Publish an event
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source component name
        """
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now(),
            source=source
        )

        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)

        # Notify sync subscribers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler {handler.__name__}: {e}")

        # Queue for async subscribers
        if event_type in self._async_subscribers:
            asyncio.create_task(self._event_queue.put(event))

        logger.debug(f"Published event {event_type.value} from {source}")

    async def _process_async_events(self):
        """Process async events in background"""
        while self._running:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                if event.type in self._async_subscribers:
                    for handler in self._async_subscribers[event.type]:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Error in async event handler {handler.__name__}: {e}")

                self._event_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing async events: {e}")

    def start(self):
        """Start the event bus"""
        self._running = True
        asyncio.create_task(self._process_async_events())
        logger.info("Event bus started")

    def stop(self):
        """Stop the event bus"""
        self._running = False
        logger.info("Event bus stopped")

    def get_event_history(self, event_type: Optional[EventType] = None, limit: int = 100) -> List[Event]:
        """Get event history
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
        """
        with self._lock:
            if event_type:
                filtered_events = [e for e in self._event_history if e.type == event_type]
            else:
                filtered_events = self._event_history.copy()

            return filtered_events[-limit:]

    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._event_history.clear()
        logger.info("Event history cleared")


# Global event bus instance
event_bus = EventBus()


def get_event_bus() -> EventBus:
    """Get the global event bus instance"""
    return event_bus


# Convenience functions for common events
def publish_jump_detected(count: int, source: str = "jump_counter"):
    """Publish jump detected event"""
    event_bus.publish(EventType.JUMP_DETECTED, {"count": count}, source)


def publish_frame_processed(fps: float, latency: float, source: str = "gui"):
    """Publish frame processed event"""
    event_bus.publish(EventType.FRAME_PROCESSED, {"fps": fps, "latency": latency}, source)


def publish_error(error: Exception, source: str = "unknown"):
    """Publish error event"""
    event_bus.publish(EventType.ERROR_OCCURRED, {"error": str(error)}, source)


def publish_performance_update(metrics: Dict[str, float], source: str = "performance_monitor"):
    """Publish performance update event"""
    event_bus.publish(EventType.PERFORMANCE_UPDATE, metrics, source)
