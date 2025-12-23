from typing import Any, Dict, List
from pydantic import BaseModel, Field
from .node import Node
from .edge import Edge, EdgeList, EdgeType
from ..agents.registry import agent_registry 
from utils.exceptions import (
    GraphBuildException, 
    CyclicGraphException,  
    NodeNotFoundException 
)
from utils.logging import get_logger
from typing import Callable, Dict, Any, List, Optional, Awaitable

# Add this new class right after NodeDefinition
class EdgeDefinition(BaseModel):
    """Serializable edge definition for workflow configurations."""
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    edge_type: str = Field(default="direct", description="Type of edge")
    condition_key: str | None = Field(default=None, description="State key for condition")
    condition_value: Any | None = Field(default=None, description="Expected value for condition")
    priority: int = Field(default=0, description="Edge priority")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class NodeDefinition(BaseModel):
    """Serializable node definition for workflow configurations."""
    
    id: str = Field(description="Unique node identifier")
    agent_id: str = Field(description="ID of the agent to execute")
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps node inputs to workflow state"
    )
    output_key: str = Field(
        default="output",
        description="Key to store output in workflow state"
    )
    skip_on_error: bool = Field(
        default=False,
        description="Whether to skip this node if it encounters an error"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    timeout: int = Field(
        default=60,
        description="Execution timeout in seconds"
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Whether to retry on failure"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Node-specific configuration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional node metadata"
    )

class WorkflowDefinition(BaseModel):
    """
    Definition of a workflow that can be converted to an executable graph.
    """
    
    workflow_id: str = Field(description="Unique workflow identifier")
    name: str = Field(description="Workflow name")
    description: str | None = None
    version: str = "1.0.0"
    
    nodes: List[NodeDefinition] = Field(description="List of nodes in the workflow")
    edges: List[EdgeDefinition] = Field(description="List of edges connecting nodes")  # CHANGED THIS LINE
    
    # Workflow configuration
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow-level configuration"
    )
    tier_requirement: str = Field(
        default="free",
        description="Minimum tier required to execute this workflow"
    )
    
    def to_executable_nodes(
        self, 
        agent_registry: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]
    ) -> List[Node]:
        """
        Convert NodeDefinitions to executable Nodes by resolving agent callables.
        
        Args:
            agent_registry: Mapping of agent_id to agent callable
            
        Returns:
            List of executable Node instances
            
        Raises:
            ValueError: If agent_id not found in registry
        """
        executable_nodes = []
        
        for node_def in self.nodes:
            agent_callable = agent_registry.get(node_def.agent_id)
            
            if not agent_callable:
                raise ValueError(
                    f"Agent '{node_def.agent_id}' not found in registry. "
                    f"Available agents: {list(agent_registry.keys())}"
                )
            
            executable_node = Node(
                id=node_def.id,
                agent_id=node_def.agent_id,
                agent_callable=agent_callable,
                input_mapping=node_def.input_mapping,
                output_key=node_def.output_key,
                skip_on_error=node_def.skip_on_error,
                max_retries=node_def.max_retries,
                timeout=node_def.timeout,
                retry_on_failure=node_def.retry_on_failure,
                config=node_def.config,
                metadata=node_def.metadata
            )
            
            executable_nodes.append(executable_node)
        
        return executable_nodes
    
    def to_executable_edges(self) -> List[Edge]:
        """Convert EdgeDefinitions to executable Edge dataclasses."""
        return [
            Edge(
                source=edge_def.source,
                target=edge_def.target,
                edge_type=EdgeType(edge_def.edge_type),
                condition_key=edge_def.condition_key,
                condition_value=edge_def.condition_value,
                priority=edge_def.priority,
                metadata=edge_def.metadata
            )
            for edge_def in self.edges
        ]
        
    class Config:
        json_schema_extra = {
            "example": {
                "workflow_id": "nutrition_plan",
                "name": "Generate Nutrition Plan",
                "description": "Calculate macros and generate meal plan",
                "nodes": [
                    {
                        "id": "calculate_macros",
                        "agent_id": "macro_calculator",
                        "input_mapping": {"weight": "input.weight"}
                    },
                    {
                        "id": "generate_meals",
                        "agent_id": "meal_planner",
                        "input_mapping": {"macros": "calculate_macros.output"}
                    }
                ],
                "edges": [
                    {"source": "calculate_macros", "target": "generate_meals"}  # UPDATED THIS
                ]
            }
        }

class Graph:
    """
    Executable graph representation of a workflow.
    """
    
    def __init__(
        self,
        workflow_id: str,
        nodes: Dict[str, Node],
        edges: EdgeList
    ):
        self.workflow_id = workflow_id
        self.nodes = nodes
        self.edges = edges
        self.logger = get_logger(f"graph.{workflow_id}")
    
    def get_node(self, node_id: str) -> Node:
        """Get a node by ID"""
        if node_id not in self.nodes:
            raise NodeNotFoundException(f"Node '{node_id}' not found in graph")
        return self.nodes[node_id]
    
    def get_start_nodes(self) -> List[str]:
        """
        Get nodes with no dependencies (root nodes).
        
        Returns:
            List of node IDs that should be executed first.
            If no edges exist, returns all nodes.
            If edges exist, returns nodes with no incoming edges (considering ALL edge types).
        """
        if not self.edges or len(self.edges) == 0:
            # No edges - all nodes are potential start nodes
            return list(self.nodes.keys())
        
        # ✅ FIX: Consider ALL edges (direct AND conditional) when determining incoming edges
        # Conditional backward edges (loops) should still count as incoming edges
        all_sources = set()
        all_targets = set()
        
        for edge in self.edges.edges:
            all_sources.add(edge.source)
            all_targets.add(edge.target)  # ← Count ALL edges, not just direct
        
        # Start nodes are those that have outgoing edges but no incoming edges
        start_nodes = list(all_sources - all_targets)
        
        if not start_nodes:
            # Graph has edges but no clear entry point
            raise GraphBuildException(
                self.workflow_id,
                "Graph has edges but no entry points (all nodes have incoming edges)"
            )
        
        return start_nodes
        
    def get_next_nodes(self, current_node: str, context: Dict[str, Any]) -> List[str]:
        """Get next nodes to execute after current node"""
        return self.edges.get_next_nodes(current_node, context)
    
    def validate(self) -> None:
        """
        Validate graph structure.
        
        Raises:
            CyclicGraphException: If graph contains cycles
            NodeNotFoundException: If node configuration is invalid
        """
        # Check for cycles
        if self.edges.has_cycle():
            raise CyclicGraphException("Graph contains a cycle")
        
        # Validate nodes
        for node_id, node in self.nodes.items():
            # Check that node ID matches
            if node.id != node_id:
                raise NodeNotFoundException(
                    f"Node ID mismatch: '{node_id}' != '{node.id}'"
                )
        
            # Validate edges reference existing nodes
            for edge in self.edges.edges:
                if edge.source not in self.nodes:  # ✓ CHANGED
                    raise NodeNotFoundException(
                        f"Edge references non-existent node: '{edge.source}'"
                    )
                if edge.target not in self.nodes:  # ✓ CHANGED
                    raise NodeNotFoundException(
                        f"Edge references non-existent node: '{edge.target}'"
                    )
    def __repr__(self) -> str:
        return f"<Graph(id={self.workflow_id}, nodes={len(self.nodes)}, edges={len(self.edges)})>"


class GraphBuilder:
    """
    Builds executable graphs from workflow definitions.
    """
    
    def __init__(self):
        self.logger = get_logger("graph_builder")
        
    def build(self, workflow: WorkflowDefinition) -> Graph:
        """
        Build an executable graph from a workflow definition.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            Executable graph
            
        Raises:
            GraphBuildException: If graph building fails
        """
        try:
            self.logger.info(
                "building_graph",
                workflow_id=workflow.workflow_id,
                node_count=len(workflow.nodes),
                edge_count=len(workflow.edges)
            )
            
            # Get agent callables from registry
            from ..agents.registry import agent_registry
            agent_callables = agent_registry.get_all_callables()
            
            # Convert NodeDefinitions to executable Nodes
            executable_nodes = workflow.to_executable_nodes(agent_callables)
            nodes = {node.id: node for node in executable_nodes}
            
            # Create edge list
            edges = EdgeList(workflow.to_executable_edges())
            
            # Create graph
            graph = Graph(
                workflow_id=workflow.workflow_id,
                nodes=nodes,
                edges=edges
            )
            
            # Validate graph structure
            graph.validate()
            
            self.logger.info(
                "graph_built_successfully",
                workflow_id=workflow.workflow_id,
                start_nodes=graph.get_start_nodes()
            )
            
            return graph
            
        except Exception as e:
            self.logger.error(
                "graph_build_failed",
                workflow_id=workflow.workflow_id,
                error=str(e)
            )
            raise GraphBuildException(workflow.workflow_id, str(e))
        
    def build_from_dict(self, data: Dict[str, Any]) -> Graph:
        """
        Build graph from dictionary representation.
        
        Args:
            data: Workflow definition as dictionary
            
        Returns:
            Executable graph
        """
        workflow = WorkflowDefinition(**data)
        return self.build(workflow)

# Global builder instance
graph_builder = GraphBuilder()