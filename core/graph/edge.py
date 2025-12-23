"""
Edge abstraction for graph execution.
Represents transitions between nodes with conditional logic.
"""

from typing import Any, Dict, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from utils.logging import get_logger


logger = get_logger(__name__)


class EdgeType(Enum):
    """Types of edges in the graph"""
    DIRECT = "direct"              # Simple A → B transition
    CONDITIONAL = "conditional"     # A → B if condition(state) == True
    MULTI = "multi"                # A → [B, C, D] based on condition result


@dataclass
class Edge:
    """
    Graph edge representing a transition between nodes.
    
    Attributes:
        source: Source node ID
        target: Target node ID (or multiple targets for MULTI type)
        condition: Optional function to determine if transition should occur
        condition_key: State key to evaluate for simple conditions
        condition_value: Expected value for condition_key
        edge_type: Type of edge (DIRECT, CONDITIONAL, MULTI)
        priority: Edge priority when multiple edges from same source (lower = higher priority)
        metadata: Additional edge metadata
    """
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECT
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    condition_key: Optional[str] = None  # Simple condition: state[key] == value
    condition_value: Optional[Any] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
        
    def can_traverse(self, state_data: Dict[str, Any]) -> bool:
        """
        Check if this edge can be traversed given current state.
        
        Args:
            state_data: Current workflow state (includes node_outputs and shared state)
            
        Returns:
            True if edge can be traversed
        """
        # Direct edges always traverse
        if self.edge_type == EdgeType.DIRECT:
            return True
        
        # Conditional edges check condition
        if self.edge_type == EdgeType.CONDITIONAL:
            # Custom condition function
            if self.condition:
                try:
                    result = self.condition(state_data)
                    logger.info(
                        f"Edge condition evaluated: {self.source} → {self.target} = {result}",
                        extra={'extra_data': {
                            'source': self.source,
                            'target': self.target,
                            'result': result
                        }}
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"Edge condition failed: {self.source} → {self.target}",
                        exc_info=e
                    )
                    return False
            
            # Simple key-value condition
            if self.condition_key and self.condition_value is not None:
                try:
                    # ✅ Check retry count in shared.input
                    retry_count = state_data.get('shared', {}).get('input', {}).get('_retry_count', 0)
                    max_retries = 1
                    
                    if retry_count >= max_retries:
                        logger.warning(
                            f"Max retries ({max_retries}) reached, blocking retry edge",
                            extra={'extra_data': {
                                'source': self.source,
                                'target': self.target,
                                'retry_count': retry_count
                            }}
                        )
                        return False
                    
                    # ✅ Look for condition_key in source node's output (at root level)
                    source_output = state_data.get(self.source, {})
                    
                    if self.condition_key not in source_output:
                        logger.warning(
                            f"Condition key not found: {self.condition_key}",
                            extra={'extra_data': {
                                'source': self.source,
                                'target': self.target,
                                'missing_key': self.condition_key,
                                'available_keys': list(source_output.keys()) if isinstance(source_output, dict) else []
                            }}
                        )
                        return False
                    
                    actual_value = source_output[self.condition_key]
                    result = actual_value == self.condition_value
                    
                    # ✅ If retry condition met, increment retry count in shared.input
                    if result and self.condition_key == 'needs_retry':
                        if 'shared' in state_data and 'input' in state_data['shared']:
                            state_data['shared']['input']['_retry_count'] = retry_count + 1
                            logger.info(
                                f"Retry triggered, count: {retry_count + 1}",
                                extra={'extra_data': {
                                    'source': self.source,
                                    'target': self.target,
                                    'retry_count': retry_count + 1
                                }}
                            )
                    
                    logger.info(
                        f"Edge condition: {self.condition_key} == {self.condition_value} → {result}",
                        extra={'extra_data': {
                            'source': self.source,
                            'target': self.target,
                            'expected': self.condition_value,
                            'actual': actual_value
                        }}
                    )
                    return result
                except KeyError as e:
                    logger.warning(
                        f"Condition key not found: {self.condition_key}",
                        extra={'extra_data': {
                            'source': self.source,
                            'target': self.target,
                            'missing_key': self.condition_key,
                            'error': str(e)
                        }}
                    )
                    return False
        
        return False
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation."""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value[key]
            else:
                raise KeyError(f"Cannot access {key} in {type(value)}")
        
        return value
    
    def __repr__(self) -> str:
        condition_str = ""
        if self.edge_type == EdgeType.CONDITIONAL:
            if self.condition:
                condition_str = " [custom condition]"
            elif self.condition_key:
                condition_str = f" [if {self.condition_key}=={self.condition_value}]"
        return f"Edge({self.source} → {self.target}{condition_str})"

@dataclass
class ConditionalEdgeGroup:
    """
    Group of edges from same source with routing logic.
    Used for switch/case style routing: A → B/C/D based on condition result.
    
    Example:
        safety_check_node → 
            - safe_node (if result == "safe")
            - review_node (if result == "needs_review")
            - stop_node (if result == "stop")
    """
    source: str
    routes: Dict[str, str]  # condition_result → target_node_id
    condition: Callable[[Dict[str, Any]], str]  # Returns which route to take
    default_target: Optional[str] = None  # Fallback if no route matches
    
    def get_target(self, state_data: Dict[str, Any]) -> Optional[str]:
        """
        Determine which target node to route to.
        
        Args:
            state_data: Current workflow state
            
        Returns:
            Target node ID or None if no valid route
        """
        try:
            condition_result = self.condition(state_data)
            
            target = self.routes.get(condition_result, self.default_target)
            
            logger.graph_transition(
                from_node=self.source,
                to_node=target or "none",
                condition=condition_result
            )
            
            return target
            
        except Exception as e:
            logger.error(
                f"Conditional routing failed for {self.source}",
                exc_info=e
            )
            return self.default_target
    
    def to_edges(self) -> List[Edge]:
        """Convert to list of conditional edges."""
        edges = []
        
        for route_value, target in self.routes.items():
            edge = Edge(
                source=self.source,
                target=target,
                edge_type=EdgeType.CONDITIONAL,
                condition=lambda state, rv=route_value: self.condition(state) == rv,
                metadata={'route_value': route_value}
            )
            edges.append(edge)
        
        # Add default edge if specified
        if self.default_target:
            default_edge = Edge(
                source=self.source,
                target=self.default_target,
                edge_type=EdgeType.CONDITIONAL,
                condition=lambda state: self.condition(state) not in self.routes,
                priority=999,  # Lowest priority
                metadata={'is_default': True}
            )
            edges.append(default_edge)
        
        return edges

@dataclass
class EdgeList:
    """
    Container for managing edges in a graph.
    Provides methods for querying and traversing edges.
    """
    edges: List[Edge] = field(default_factory=list)
    conditional_groups: List[ConditionalEdgeGroup] = field(default_factory=list)
    
    def add_edge(self, edge: Edge) -> None:
        """Add a single edge to the list."""
        self.edges.append(edge)
    
    def add_edges(self, edges: List[Edge]) -> None:
        """Add multiple edges to the list."""
        self.edges.extend(edges)
    
    def add_conditional_group(self, group: ConditionalEdgeGroup) -> None:
        """Add a conditional edge group and convert to edges."""
        self.conditional_groups.append(group)
        self.edges.extend(group.to_edges())
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """
        Get all edges originating from a node.
        
        Args:
            node_id: Source node ID
            
        Returns:
            List of edges from this node, sorted by priority
        """
        outgoing = [e for e in self.edges if e.source == node_id]
        return sorted(outgoing, key=lambda e: e.priority)
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """
        Get all edges pointing to a node.
        
        Args:
            node_id: Target node ID
            
        Returns:
            List of edges to this node
        """
        return [e for e in self.edges if e.target == node_id]
    
    def get_next_nodes(self, current_node_id: str, state_data: Dict[str, Any]) -> List[str]:
        """
        Get list of next node IDs that should be executed based on current state.
        
        Args:
            current_node_id: Current node ID
            state_data: Current workflow state
            
        Returns:
            List of node IDs to execute next
        """
        outgoing = self.get_outgoing_edges(current_node_id)
        
        if not outgoing:
            return []
        
        next_nodes = []
        
        for edge in outgoing:
            if edge.can_traverse(state_data):
                next_nodes.append(edge.target)
                
                # For conditional edges, typically only take first match
                if edge.edge_type == EdgeType.CONDITIONAL:
                    break
        
        return next_nodes
    
    def has_outgoing_edges(self, node_id: str) -> bool:
        """Check if node has any outgoing edges."""
        return any(e.source == node_id for e in self.edges)
    
    def has_incoming_edges(self, node_id: str) -> bool:
        """Check if node has any incoming edges."""
        return any(e.target == node_id for e in self.edges)
    
    def get_source_nodes(self) -> List[str]:
        """
        Get all nodes with no incoming edges (entry points).
        
        Returns:
            List of node IDs that are entry points
        """
        all_sources = {e.source for e in self.edges}
        all_targets = {e.target for e in self.edges}
        return list(all_sources - all_targets)
    
    def get_sink_nodes(self) -> List[str]:
        """
        Get all nodes with no outgoing edges (exit points).
        
        Returns:
            List of node IDs that are exit points
        """
        all_sources = {e.source for e in self.edges}
        all_targets = {e.target for e in self.edges}
        return list(all_targets - all_sources)
    
    def has_cycle(self) -> bool:
        """
        Check if the graph has cycles using depth-first search.
        Only checks DIRECT edges - conditional edges are allowed to create loops.
        
        Returns:
            True if a cycle is detected in the direct edge path, False otherwise
        """
        # Build adjacency list from DIRECT edges only
        # Conditional edges are allowed to create loops for retry/rerun logic
        graph: Dict[str, List[str]] = {}
        all_nodes: set[str] = set()
        
        for edge in self.edges:
            # Skip conditional edges - they're allowed to create loops
            if edge.edge_type == EdgeType.CONDITIONAL:
                # Still add nodes to the set for completeness
                all_nodes.add(edge.source)
                all_nodes.add(edge.target)
                continue
            
            all_nodes.add(edge.source)
            all_nodes.add(edge.target)
            
            if edge.source not in graph:
                graph[edge.source] = []
            graph[edge.source].append(edge.target)
        
        # Track visited nodes and recursion stack
        visited: set[str] = set()
        rec_stack: set[str] = set()
        
        def dfs(node: str) -> bool:
            """DFS helper to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            
            # Check all neighbors
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found a back edge - cycle detected
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes (handles disconnected components)
        for node in all_nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False
    
    def validate(self, node_ids: set[str]) -> List[str]:
        """
        Validate that all edges reference valid nodes.
        
        Args:
            node_ids: Set of valid node IDs in the graph
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        for edge in self.edges:
            if edge.source not in node_ids:
                errors.append(f"Edge references non-existent source node: {edge.source}")
            if edge.target not in node_ids:
                errors.append(f"Edge references non-existent target node: {edge.target}")
        
        return errors
    
    def __len__(self) -> int:
        return len(self.edges)
    
    def __iter__(self):
        return iter(self.edges)
    
    def __repr__(self) -> str:
        return f"EdgeList({len(self.edges)} edges)"
    
def create_edge(
    source: str,
    target: str,
    condition: Optional[Callable] = None,
    condition_key: Optional[str] = None,
    condition_value: Optional[Any] = None,
    priority: int = 0,
    **metadata
) -> Edge:
    """
    Factory function to create an Edge.
    
    Args:
        source: Source node ID
        target: Target node ID
        condition: Custom condition function
        condition_key: State key for simple condition
        condition_value: Expected value for condition_key
        priority: Edge priority (lower = higher priority)
        **metadata: Additional metadata
        
    Returns:
        Configured Edge instance
    """
    edge_type = EdgeType.DIRECT
    
    if condition or (condition_key and condition_value is not None):
        edge_type = EdgeType.CONDITIONAL
    
    return Edge(
        source=source,
        target=target,
        edge_type=edge_type,
        condition=condition,
        condition_key=condition_key,
        condition_value=condition_value,
        priority=priority,
        metadata=metadata
    )


def create_conditional_routes(
    source: str,
    routes: Dict[str, str],
    condition: Callable[[Dict[str, Any]], str],
    default_target: Optional[str] = None
) -> ConditionalEdgeGroup:
    """
    Create a conditional edge group for routing logic.
    
    Example:
        routes = {
            "safe": "generate_plan_node",
            "needs_review": "review_node",
            "stop": "end_node"
        }
        
        def check_safety(state):
            result = state["node_outputs"]["safety_check"]["result"]
            return result  # Returns "safe", "needs_review", or "stop"
        
        edge_group = create_conditional_routes(
            source="safety_check_node",
            routes=routes,
            condition=check_safety
        )
    
    Args:
        source: Source node ID
        routes: Mapping of condition results to target nodes
        condition: Function that returns route key
        default_target: Fallback target if no route matches
        
    Returns:
        ConditionalEdgeGroup instance
    """
    return ConditionalEdgeGroup(
        source=source,
        routes=routes,
        condition=condition,
        default_target=default_target
    )


# Common condition helpers
def state_equals(key: str, value: Any) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a condition that checks if state[key] == value.
    
    Example:
        edge = Edge(
            source="node_a",
            target="node_b",
            condition=state_equals("status", "success")
        )
    """
    def condition(state: Dict[str, Any]) -> bool:
        try:
            keys = key.split('.')
            val = state
            for k in keys:
                val = val[k]
            return val == value
        except (KeyError, TypeError):
            return False
    return condition


def node_output_equals(node_id: str, key: str, value: Any) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a condition that checks node output value.
    
    Example:
        edge = Edge(
            source="safety_check",
            target="generate_plan",
            condition=node_output_equals("safety_check", "is_safe", True)
        )
    """
    def condition(state: Dict[str, Any]) -> bool:
        try:
            output = state["node_outputs"][node_id]
            return output.get(key) == value
        except (KeyError, TypeError):
            return False
    return condition


def all_conditions(*conditions: Callable) -> Callable[[Dict[str, Any]], bool]:
    """
    Combine multiple conditions with AND logic.
    
    Example:
        edge = Edge(
            source="node_a",
            target="node_b",
            condition=all_conditions(
                state_equals("tier", "premium"),
                node_output_equals("validation", "passed", True)
            )
        )
    """
    def combined(state: Dict[str, Any]) -> bool:
        return all(cond(state) for cond in conditions)
    return combined


def any_condition(*conditions: Callable) -> Callable[[Dict[str, Any]], bool]:
    """
    Combine multiple conditions with OR logic.
    """
    def combined(state: Dict[str, Any]) -> bool:
        return any(cond(state) for cond in conditions)
    return combined
