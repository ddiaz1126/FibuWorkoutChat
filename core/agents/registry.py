"""
Agent registry for discovering and managing agents.
"""

"""
Agent registry for discovering and managing agents.
"""

from typing import Dict, Optional, Type, Callable, Any, Awaitable
from .base import BaseAgent
from utils.exceptions import AgentNotFoundException, AgentRegistrationException
from utils.logging import get_logger


class AgentRegistry:
    """
    Central registry for all agents in the system.
    
    Features:
    - Agent registration and discovery
    - Singleton pattern for global access
    - Type validation
    - Lifecycle management
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
        self.logger = get_logger("agent_registry")
    
    def register(self, agent_id: str, agent: BaseAgent) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to register
            
        Raises:
            AgentRegistrationException: If registration fails
        """
        if not isinstance(agent, BaseAgent):
            raise AgentRegistrationException(
                agent_id,
                f"Agent must be an instance of BaseAgent, got {type(agent)}"
            )
        
        if agent_id in self._agents:
            self.logger.warning(
                "agent_already_registered",
                agent_id=agent_id,
                message="Overwriting existing agent"
            )
        
        self._agents[agent_id] = agent
        self.logger.info(
            "agent_registered",
            agent_id=agent_id,
            agent_type=type(agent).__name__
        )
    
    def register_class(self, agent_id: str, agent_class: Type[BaseAgent], **kwargs) -> None:
        """
        Register an agent by instantiating its class.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_class: Agent class to instantiate
            **kwargs: Arguments to pass to agent constructor
            
        Raises:
            AgentRegistrationException: If registration fails
        """
        try:
            agent = agent_class(**kwargs)
            self.register(agent_id, agent)
        except Exception as e:
            raise AgentRegistrationException(
                agent_id,
                f"Failed to instantiate agent: {str(e)}"
            )
    
    def get(self, agent_id: str) -> BaseAgent:
        """
        Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance
            
        Raises:
            AgentNotFoundException: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentNotFoundException(agent_id)
        
        return self._agents[agent_id]
    
    def get_callable(self, agent_id: str) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
        """
        Get an agent's execute method as a callable.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent's execute method
            
        Raises:
            AgentNotFoundException: If agent not found
        """
        agent = self.get(agent_id)
        return agent.execute
    
    def get_all_callables(self) -> Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
        """
        Get all agents as a dictionary of callables.
        
        Returns:
            Dictionary mapping agent_id to agent execute method
        """
        return {
            agent_id: agent.execute
            for agent_id, agent in self._agents.items()
        }
    
    def has(self, agent_id: str) -> bool:
        """
        Check if an agent exists in the registry.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent exists, False otherwise
        """
        return agent_id in self._agents
    
    def unregister(self, agent_id: str) -> None:
        """
        Remove an agent from the registry.
        
        Args:
            agent_id: Agent identifier
            
        Raises:
            AgentNotFoundException: If agent not found
        """
        if agent_id not in self._agents:
            raise AgentNotFoundException(agent_id)
        
        del self._agents[agent_id]
        self.logger.info("agent_unregistered", agent_id=agent_id)
    
    def list_agents(self) -> Dict[str, str]:
        """
        List all registered agents.
        
        Returns:
            Dictionary mapping agent_id to agent type name
        """
        return {
            agent_id: type(agent).__name__
            for agent_id, agent in self._agents.items()
        }
    
    def clear(self) -> None:
        """Clear all agents from the registry."""
        self._agents.clear()
        self.logger.info("agent_registry_cleared")
    
    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self._agents)
    
    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent exists using 'in' operator."""
        return agent_id in self._agents


# Global singleton instance
agent_registry = AgentRegistry()