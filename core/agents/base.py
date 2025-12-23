"""
Base agent interface for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .models import AgentInput, AgentOutput



class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement the run() method which takes AgentInput
    and returns AgentOutput.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the agent.
        
        Args:
            name: Optional name for the agent
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    async def run(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the agent's primary logic.
        
        Args:
            input_data: Input data and context for the agent
            
        Returns:
            AgentOutput containing results or errors
        """
        pass
    
    async def execute(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with dictionary input/output.
        This is the method called by the workflow engine.
        
        Args:
            input_dict: Input data as dictionary
            
        Returns:
            Output data as dictionary (full AgentOutput)
        """
        print(f"[BaseAgent.execute] Agent={self.name} raw input_dict keys={list(input_dict.keys())}")
        print(f"[BaseAgent.execute] Agent={self.name} raw input_dict={input_dict!r}")

        # Convert dict to AgentInput
        agent_input = AgentInput(data=input_dict)
        print(f"[BaseAgent.execute] Agent={self.name} AgentInput.data keys={list(agent_input.data.keys())}")

        # ✅ Await validation (supports async overrides)
        is_valid = await self.validate_input(agent_input)
        if not is_valid:
            raise ValueError(f"Invalid input for agent {self.name}")
        
        # Call the abstract run method
        agent_output = await self.run(agent_input)
        
        # If agent reports failure, raise to let workflow/graph handle it
        if not agent_output.success:
            raise Exception(agent_output.error or "Agent execution failed")
        
        # ✅ Return the full AgentOutput as dict so the graph has everything
        return agent_output.model_dump()
    
    async def validate_input(self, input_data: AgentInput) -> bool:
        """
        Validate input data before execution.
        
        Args:
            input_data: Input to validate
            
        Returns:
            True if valid, False otherwise
        """
        return True
    
    def get_required_inputs(self) -> list[str]:
        """
        Get list of required input keys.
        
        Returns:
            List of required input field names
        """
        return []
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and metadata.
        
        Returns:
            Dictionary describing agent capabilities
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "required_inputs": self.get_required_inputs()
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata including schemas"""
        return {
            "agent_id": getattr(self.__class__, 'agent_id', self.__class__.__name__),
            "name": getattr(self.__class__, 'name', self.name),
            "description": getattr(self.__class__, 'description', ''),
            "version": getattr(self.__class__, 'version', '1.0.0'),
            "category": getattr(self.__class__, 'category', 'general'),
            "tags": getattr(self.__class__, 'tags', []),
            "input_schema": getattr(self.__class__, 'input_schema', {}),   # ← Changed
            "output_schema": getattr(self.__class__, 'output_schema', {})  # ← Changed
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"