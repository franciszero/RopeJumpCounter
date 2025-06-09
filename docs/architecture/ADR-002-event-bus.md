# ADR-002: Event Bus System

## Status
Accepted

## Context
As the RopeJumpCounter application grows, components need to communicate with each other without creating tight coupling. The current approach of direct method calls and callbacks creates complex dependency graphs and makes the system difficult to maintain and extend.

## Decision
We will implement an event bus system to enable decoupled communication between application components through event publishing and subscription.

## Consequences

### Positive
- **Decoupled Communication**: Components communicate through events without direct dependencies
- **Extensibility**: New components can easily subscribe to existing events
- **Asynchronous Support**: Events can be processed asynchronously
- **Event History**: Built-in event history for debugging and monitoring
- **Flexible Architecture**: Easy to add new event types and handlers

### Negative
- **Event Ordering**: Events may not be processed in the exact order they were published
- **Debugging Complexity**: Event flow can be harder to trace than direct method calls
- **Memory Usage**: Event history storage requires additional memory
- **Learning Curve**: Team needs to understand event-driven patterns

## Implementation Details

### Event Types
```python
class EventType(Enum):
    JUMP_DETECTED = "jump_detected"
    FRAME_PROCESSED = "frame_processed"
    MODEL_LOADED = "model_loaded"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_UPDATE = "performance_update"
```

### Event Publishing
```python
# Publish events
event_bus.publish(EventType.JUMP_DETECTED, {"count": 5}, "jump_counter")
event_bus.publish(EventType.PERFORMANCE_UPDATE, metrics, "performance_monitor")
```

### Event Subscription
```python
# Subscribe to events
event_bus.subscribe(EventType.JUMP_DETECTED, on_jump_detected)
event_bus.subscribe(EventType.ERROR_OCCURRED, on_error, async_handler=True)
```

### Event Structure
```python
@dataclass
class Event:
    type: EventType
    data: Any
    timestamp: datetime
    source: str
```

## Usage Patterns

### Synchronous Events
For immediate processing that doesn't block the main thread:
```python
def on_jump_detected(event):
    update_display(event.data['count'])
```

### Asynchronous Events
For operations that might take time or need to be queued:
```python
async def on_performance_update(event):
    await save_metrics_to_database(event.data)
```

## Migration Strategy
1. Identify existing direct component communications
2. Define appropriate event types for each communication pattern
3. Replace direct calls with event publishing
4. Create event handlers for existing functionality
5. Gradually migrate components to use the event bus

## Related ADRs
- ADR-001: Dependency Injection Container
- ADR-003: Plugin Architecture 