"""
Node abstraction for graph execution.
Represents a single executable unit in the workflow graph.
"""

from typing import Any, Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from utils.logging import get_logger
from ..agents.registry import agent_registry  # Add this import


logger = get_logger(__name__)


class NodeStatus(Enum):
    """Node execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result of node execution"""
    node_id: str
    status: NodeStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate execution duration in milliseconds"""
        if self.started_at and self.completed_at:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None


@dataclass
class Node:
    """
    Graph node representing an agent execution step.
    
    Attributes:
        id: Unique node identifier
        agent_id: ID of the agent to execute
        agent_callable: The actual agent function to call
        input_mapping: How to map state data to agent input
        output_key: Where to store the output in state
        condition: Optional condition function to determine if node should run
        metadata: Additional node metadata
    """
    id: str
    agent_id: str
    agent_callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    input_mapping: Dict[str, str] = field(default_factory=dict)
    output_key: str = "output"
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    skip_on_error: bool = False  # ADD THIS LINE
    max_retries: int = 3  # ADD THIS
    timeout: int = 60  # ADD THIS
    retry_on_failure: bool = True  # ADD THIS
    config: Dict[str, Any] = field(default_factory=dict) 
    metadata: Dict[str, Any] = field(default_factory=dict)

    def should_execute(self, state_data: Dict[str, Any]) -> bool:
        """
        Determine if this node should execute.
        
        Args:
            state_data: Current workflow state
            
        Returns:
            True if node should execute
        """
        return self.can_execute(state_data)
    
    def resolve_inputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve inputs from context using input_mapping.
        
        Args:
            context: Execution context
            
        Returns:
            Resolved input dictionary
        """
        return self._map_input(context)
    
    async def execute(self, state_data: Dict[str, Any]) -> NodeResult:
        """
        Execute the node's agent with mapped input from state.
        
        Args:
            state_data: Current workflow state
            
        Returns:
            NodeResult with output or error
        """
        result = NodeResult(
            node_id=self.id,
            status=NodeStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        logger.agent_start(self.agent_id, state_data)
        
        try:
            # Check condition if present
            if self.condition and not self.condition(state_data):
                logger.info(f"Node {self.id} skipped due to condition")
                result.status = NodeStatus.SKIPPED
                result.completed_at = datetime.utcnow()
                return result
            
            # Map input from state
            agent_input = self._map_input(state_data)
            
            logger.info(
                f"Executing node {self.id} with agent {self.agent_id}",
                extra={'extra_data': {
                    'node_id': self.id,
                    'agent_id': self.agent_id,
                    'input_keys': list(agent_input.keys())
                }}
            )
            
            # Execute agent
            output = await self.agent_callable(agent_input)
            
            # Store result
            result.status = NodeStatus.COMPLETED
            result.output = output
            result.completed_at = datetime.utcnow()
            
            logger.agent_complete(
                self.agent_id,
                duration_ms=result.duration_ms or 0,
                output_size=len(str(output))
            )
            
            return result
            
        except Exception as e:
            result.status = NodeStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.utcnow()
            
            logger.agent_error(self.agent_id, e)
            
            return result
    
    def _map_input(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map state data to agent input using input_mapping.
        
        input_mapping format:
        {
            "agent_param": "state.path.to.value",
            "weight": "input.weight",
            "macros": "node_outputs.calculate_macros.protein"
        }
        
        Args:
            state_data: Current state
            
        Returns:
            Mapped input dictionary for agent
        """
        if not self.input_mapping:
            return state_data
        
        mapped = {}
        
        for agent_key, state_path in self.input_mapping.items():
            try:
                value = self._get_nested_value(state_data, state_path)
                mapped[agent_key] = value
            except KeyError as e:
                logger.warning(
                    f"Input mapping failed for {agent_key}: {state_path}",
                    extra={'extra_data': {
                        'node_id': self.id,
                        'missing_path': state_path
                    }}
                )
                # Continue without this value - let agent handle missing input
                
        return mapped
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get value from nested dict using dot notation.
        
        Examples:
            _get_nested_value({"a": {"b": 1}}, "a.b") -> 1
            _get_nested_value({"input": {"weight": 70}}, "input.weight") -> 70
        """
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Cannot access {key} in {type(value)}")
        
        return value
    
    def can_execute(self, state_data: Dict[str, Any]) -> bool:
        """
        Check if node can execute based on:
        1. Required inputs are available in state
        2. Condition passes (if present)
        
        Args:
            state_data: Current state
            
        Returns:
            True if node can execute
        """
        # Check condition
        if self.condition and not self.condition(state_data):
            return False
        
        # Get the agent to check which fields are required
        agent = agent_registry.get(self.agent_id)
        required_fields = []
        
        if agent and hasattr(agent, 'input_schema'):
            required_fields = agent.input_schema.get("required", [])
        
        # Only check required inputs are available
        for agent_key, state_path in self.input_mapping.items():
            # Skip optional fields
            if agent_key not in required_fields:
                continue
                
            try:
                self._get_nested_value(state_data, state_path)
            except KeyError:
                logger.warning(
                    f"Node {self.id} cannot execute - missing REQUIRED input: {state_path}"
                )
                return False
        
        return True

def create_node(
    node_id: str,
    agent_id: str,
    agent_callable: Callable,
    input_mapping: Optional[Dict[str, str]] = None,
    output_key: str = "output",
    condition: Optional[Callable] = None,
    **metadata
) -> Node:
    """
    Factory function to create a Node.
    
    Args:
        node_id: Unique identifier
        agent_id: Agent to execute
        agent_callable: Agent function
        input_mapping: State to agent input mapping
        output_key: Where to store output
        condition: Optional execution condition
        **metadata: Additional metadata
        
    Returns:
        Configured Node instance
    """
    return Node(
        id=node_id,
        agent_id=agent_id,
        agent_callable=agent_callable,
        input_mapping=input_mapping or {},
        output_key=output_key,
        condition=condition,
        metadata=metadata
    )