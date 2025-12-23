import asyncio
from typing import Any, Dict, List
from datetime import datetime
from .builder import Graph
from .node import Node
from .state import GraphState, ExecutionStatus, NodeStatus
from ..agents.registry import agent_registry
from ..agents.models import AgentInput
from utils.exceptions import (
    GraphExecutionException,
    AgentNotFoundException,
    AgentTimeoutException
)
from utils.logging import get_logger


class GraphExecutor:
    """
    Executes graphs by traversing nodes and running agents.
    
    Features:
    - Parallel execution of independent nodes
    - Retry logic for failed nodes
    - Timeout handling
    - State management
    - Conditional branching
    """
    
    def __init__(self, max_retries: int = 3, default_timeout: int = 60):
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self.logger = get_logger("graph_executor")
    
    async def execute(
        self,
        graph: Graph,
        initial_data: Dict[str, Any],
        workflow_id: str,
        user_id: str | None = None,
        tier: str = "free"
    ) -> GraphState:
        """
        Execute a graph workflow.
        
        Args:
            graph: Graph to execute
            initial_data: Initial input data
            workflow_id: Unique workflow execution ID
            user_id: User ID for context
            tier: User tier for permissions
            
        Returns:
            Final graph state with results
        """
        # Initialize state
        state = GraphState(
            workflow_id=workflow_id,
            graph_id=graph.workflow_id,
            user_id=user_id,
            tier=tier,
            max_retries=self.max_retries,
            shared_context=initial_data
        )
        
        # Add all nodes to state
        for node_id, node in graph.nodes.items():
            state.add_node(node_id, node.agent_id)
        
        try:
            self.logger.info(
                "graph_execution_started",
                workflow_id=workflow_id,
                graph_id=graph.workflow_id,
                node_count=len(graph.nodes)
            )
            
            state.start_execution()
            
            # Get starting nodes
            current_nodes = graph.get_start_nodes()
            
            # Execute graph using BFS-style traversal
            while current_nodes:
                # Execute current batch of nodes in parallel
                next_nodes = await self._execute_node_batch(
                    graph, state, current_nodes
                )
                
                # Check for execution failure
                if state.status == ExecutionStatus.FAILED:
                    break
                
                # Move to next batch
                current_nodes = list(set(next_nodes))  # Remove duplicates
            
            # Mark as completed if not already failed
            if state.status != ExecutionStatus.FAILED:
                state.complete_execution()
                self.logger.info(
                    "graph_execution_completed",
                    workflow_id=workflow_id,
                    execution_time=state.get_execution_time()
                )
            
        except Exception as e:
            error_msg = f"Graph execution failed: {str(e)}"
            state.fail_execution(error_msg)
            self.logger.error(
                "graph_execution_error",
                workflow_id=workflow_id,
                error=str(e)
            )
        
        return state
    
    async def _execute_node_batch(
        self,
        graph: Graph,
        state: GraphState,
        node_ids: List[str]
    ) -> List[str]:
        """
        Execute a batch of nodes in parallel.
        
        Args:
            graph: Graph being executed
            state: Current execution state
            node_ids: List of node IDs to execute
            
        Returns:
            List of next node IDs to execute
        """
        # Execute nodes in parallel
        tasks = [
            self._execute_node(graph, state, node_id)
            for node_id in node_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect next nodes from all successful executions
        next_nodes = []
        for node_id, result in zip(node_ids, results):
            if isinstance(result, Exception):
                self.logger.error(
                    "node_execution_exception",
                    node_id=node_id,
                    error=str(result)
                )
                state.fail_node(node_id, str(result))
                state.fail_execution(f"Node {node_id} failed: {str(result)}")
                return []
            elif result:
                next_nodes.extend(result)
        
        return next_nodes
    
    async def _execute_node(
        self,
        graph: Graph,
        state: GraphState,
        node_id: str
    ) -> List[str]:
        """
        Execute a single node with retry logic.
        
        Args:
            graph: Graph being executed
            state: Current execution state
            node_id: Node ID to execute
            
        Returns:
            List of next node IDs to execute
        """
        node = graph.get_node(node_id)
        
        # Check if node should be skipped
        if node.skip_on_error and state.status == ExecutionStatus.FAILED:
            state.skip_node(node_id)
            return []
        
        # Check conditional execution
        context = self._build_context(state)
        if not node.should_execute(context):
            self.logger.info(
                "node_skipped_by_condition",
                node_id=node_id,
                condition=node.condition
            )
            state.skip_node(node_id)
            return graph.get_next_nodes(node_id, context)
        
        # Execute with retries
        for attempt in range(node.max_retries + 1):
            try:
                state.start_node(node_id)
                
                # Get agent
                agent = agent_registry.get(node.agent_id)
                
                # Prepare input
                agent_input = self._prepare_agent_input(node, state)
                
                # Execute with timeout
                output = await asyncio.wait_for(
                    agent.run(agent_input),
                    timeout=node.timeout
                )
                
                # Check if execution was successful
                if output.success:
                    state.complete_node(node_id, output.data)

                    state.store_node_metadata(node_id, output.metadata)  # ✅ NEW: separate call
                    
                    # Store output in shared context if key specified
                    if node.output_key:
                        state.shared_context[node.output_key] = output.data
                    
                    # ✅ GENERIC: If output contains needs_retry, inject ai_* parameters
                    if output.data.get("needs_retry") == True:
                        # Ensure 'input' exists in shared_context
                        if 'input' not in state.shared_context:
                            state.shared_context['input'] = {}
                        
                        # Inject any ai_* parameters from output into shared.input
                        for key, value in output.data.items():
                            if key.startswith('ai_'):
                                state.shared_context['input'][key] = value
                                self.logger.info(
                                    "retry_parameter_injected",
                                    node_id=node_id,
                                    param=key,
                                    value=value
                                )
                    
                    # Get next nodes
                    context = self._build_context(state)
                    next_nodes_list = graph.get_next_nodes(node_id, context)
                    print(f"🔍 [EXECUTOR DEBUG] Node '{node_id}' → Next nodes: {next_nodes_list}")
                    return next_nodes_list
                else:
                    # Agent execution failed
                    if attempt < node.max_retries and node.retry_on_failure:
                        state.increment_retry(node_id)
                        self.logger.warning(
                            "node_execution_failed_retrying",
                            node_id=node_id,
                            attempt=attempt + 1,
                            error=output.error
                        )
                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        state.fail_node(node_id, output.error or "Unknown error")
                        return []
                        
            except asyncio.TimeoutError:
                error_msg = f"Node execution timeout after {node.timeout}s"
                if attempt < node.max_retries and node.retry_on_failure:
                    state.increment_retry(node_id)
                    self.logger.warning(
                        "node_timeout_retrying",
                        node_id=node_id,
                        attempt=attempt + 1
                    )
                    continue
                else:
                    state.fail_node(node_id, error_msg)
                    return []
                    
            except AgentNotFoundException as e:
                state.fail_node(node_id, str(e))
                return []
                
            except Exception as e:
                error_msg = f"Node execution error: {str(e)}"
                if attempt < node.max_retries and node.retry_on_failure:
                    state.increment_retry(node_id)
                    self.logger.warning(
                        "node_execution_error_retrying",
                        node_id=node_id,
                        attempt=attempt + 1,
                        error=str(e)
                    )
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    state.fail_node(node_id, error_msg)
                    return []
        
        return []
    
    def _prepare_agent_input(self, node: Node, state: GraphState) -> AgentInput:
        """
        Prepare input for agent execution.
        
        Args:
            node: Node being executed
            state: Current execution state
            
        Returns:
            AgentInput for the agent
        """
        # Build context
        context = self._build_context(state)
        
        # Resolve inputs from context
        input_data = node.resolve_inputs(context)
        
        # Create agent input
        return AgentInput(
            data=input_data,
            context={
                "workflow_id": state.workflow_id,
                "user_id": state.user_id,
                "tier": state.tier,
                "node_id": node.id,
                **state.shared_context
            },
            config=node.config
        )
    
    def _build_context(self, state: GraphState) -> Dict[str, Any]:
        """
        Build execution context from state.
        
        Args:
            state: Current execution state
            
        Returns:
            Context dictionary
        """
        context = {
            "shared": state.shared_context,
            **state.node_outputs
        }
        return context


# Global executor instance
graph_executor = GraphExecutor()