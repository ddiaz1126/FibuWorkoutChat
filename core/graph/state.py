from typing import Any, Dict, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class ExecutionStatus(str, Enum):
    """Status of workflow execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(str, Enum):
    """Status of individual node execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class NodeState(BaseModel):
    """State of a single node in the graph"""
    
    node_id: str
    agent_id: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphState(BaseModel):
    """
    Complete state of a graph execution.
    
    Tracks:
    - Overall execution status
    - Individual node states
    - Data flow between nodes
    - Execution timeline
    """
    
    workflow_id: str = Field(description="Unique workflow execution ID")
    graph_id: str = Field(description="Graph template ID")
    status: ExecutionStatus = ExecutionStatus.PENDING
    
    # Node tracking
    nodes: Dict[str, NodeState] = Field(
        default_factory=dict,
        description="State of each node"
    )
    current_node: str | None = Field(
        default=None,
        description="Currently executing node"
    )
    completed_nodes: Set[str] = Field(
        default_factory=set,
        description="Set of completed node IDs"
    )
    
    # Data flow
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared data accessible to all nodes"
    )
    node_outputs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Output data from each node"
    )
    
    # Execution metadata
    user_id: str | None = None
    tier: str = "free"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    
    # Configuration
    max_retries: int = 3
    timeout: int = 300
    
    class Config:
        use_enum_values = True
    
    def add_node(self, node_id: str, agent_id: str) -> None:
        """Add a node to the state"""
        self.nodes[node_id] = NodeState(
            node_id=node_id,
            agent_id=agent_id
        )
    
    def start_execution(self) -> None:
        """Mark execution as started"""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete_execution(self) -> None:
        """Mark execution as completed"""
        self.status = ExecutionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def fail_execution(self, error: str) -> None:
        """Mark execution as failed"""
        self.status = ExecutionStatus.FAILED
        self.error = error
        self.completed_at = datetime.utcnow()
    
    def start_node(self, node_id: str) -> None:
        """Mark node as started"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        self.current_node = node_id
        self.nodes[node_id].status = NodeStatus.RUNNING
        self.nodes[node_id].started_at = datetime.utcnow()
    
    def complete_node(self, node_id: str, output_data: Dict[str, Any]) -> None:
        """Mark node as completed"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        self.nodes[node_id].status = NodeStatus.COMPLETED
        self.nodes[node_id].output_data = output_data
        self.nodes[node_id].completed_at = datetime.utcnow()
        self.completed_nodes.add(node_id)
        self.node_outputs[node_id] = output_data
    
    def fail_node(self, node_id: str, error: str) -> None:
        """Mark node as failed"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        self.nodes[node_id].status = NodeStatus.FAILED
        self.nodes[node_id].error = error
        self.nodes[node_id].completed_at = datetime.utcnow()
    
    def skip_node(self, node_id: str) -> None:
        """Mark node as skipped"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        self.nodes[node_id].status = NodeStatus.SKIPPED
        self.completed_nodes.add(node_id)
    
    def increment_retry(self, node_id: str) -> None:
        """Increment retry count for a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        self.nodes[node_id].retry_count += 1
    
    def get_node_output(self, node_id: str) -> Dict[str, Any]:
        """Get output data from a node"""
        return self.node_outputs.get(node_id, {})
    
    def is_node_completed(self, node_id: str) -> bool:
        """Check if node is completed"""
        return node_id in self.completed_nodes
    
    def get_execution_time(self) -> float | None:
        """Get total execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    # In GraphState class, ADD a new method:
    def store_node_metadata(self, node_id: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a node separately"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found in state")
        
        if metadata:
            self.nodes[node_id].metadata = metadata