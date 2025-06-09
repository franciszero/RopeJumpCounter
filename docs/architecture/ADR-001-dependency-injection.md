# ADR-001: Dependency Injection Container

## Status
Accepted

## Context
The RopeJumpCounter application has grown in complexity with multiple components that need to be managed and coordinated. The current approach of direct instantiation and tight coupling between components makes the code difficult to test, maintain, and extend.

## Decision
We will implement a dependency injection container to manage application dependencies and promote loose coupling between components.

## Consequences

### Positive
- **Improved Testability**: Components can be easily mocked and tested in isolation
- **Loose Coupling**: Components depend on abstractions rather than concrete implementations
- **Centralized Configuration**: All service configuration is managed in one place
- **Easier Maintenance**: Changes to component dependencies are centralized
- **Better Error Handling**: Centralized error handling and logging

### Negative
- **Increased Complexity**: Additional abstraction layer
- **Learning Curve**: Team needs to understand DI patterns
- **Performance Overhead**: Minimal overhead from container management

## Implementation Details

### Container Structure
```python
class Container:
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._config: Optional[AppConfig] = None
        self._state: Optional[AppState] = None
```

### Service Registration
```python
# Register configuration
container.register_config(config)

# Register singleton services
container.register_singleton('predictor', VideoPredictor, model_path)

# Register service instances
container.register_service('logger', logger)
```

### Service Retrieval
```python
# Get services
predictor = container.get_service('predictor')
config = container.get_config()
state = container.get_state()
```

## Migration Strategy
1. Create the container infrastructure
2. Gradually migrate existing components to use the container
3. Update tests to use mocked services from the container
4. Remove direct instantiation in favor of container-based dependency resolution

## Related ADRs
- ADR-002: Event Bus System
- ADR-003: Plugin Architecture 