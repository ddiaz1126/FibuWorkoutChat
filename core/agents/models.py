from typing import Any, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

class AgentInput(BaseModel):
    """Input model for agent execution"""
    
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Input data for the agent"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution context (user_id, workflow_id, etc.)"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent-specific configuration"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {"user_input": "Calculate my macros"},
                "context": {"user_id": "123", "workflow_id": "wf_456"},
                "config": {"model": "gpt-4", "temperature": 0.7}
            }
        }

class AgentOutput(BaseModel):
    """Output model for agent execution"""
    
    success: bool = Field(
        description="Whether the agent execution was successful"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output data from the agent"
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata (timing, tokens used, etc.)"
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="List of artifacts generated (file paths, URLs, etc.)"
    )
    
    @staticmethod
    def _convert_numpy(obj):
        """Convert NumPy types to native Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: AgentOutput._convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [AgentOutput._convert_numpy(item) for item in obj]
        return obj
    
    def model_post_init(self, __context):
        """Automatically convert NumPy types after initialization"""
        self.data = self._convert_numpy(self.data)
        self.metadata = self._convert_numpy(self.metadata)
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"macros": {"protein": 150, "carbs": 200, "fat": 60}},
                "error": None,
                "metadata": {"processing_time": 1.2, "tokens_used": 250},
                "artifacts": []
            }
        }

class AgentMetadata(BaseModel):
    """Metadata about an agent"""
    
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    description: str = Field(description="Agent description")
    version: str = Field(description="Agent version")
    category: str = Field(
        default="general",
        description="Agent category (nutrition, workout, coaching, etc.)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Agent tags for discovery"
    )
    author: str | None = Field(
        default=None,
        description="Agent author"
    )
    tier_requirement: str = Field(
        default="free",
        description="Minimum tier required to use this agent"
    )
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for agent input validation"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema for agent output"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "macro_calculator",
                "name": "Macro Calculator",
                "description": "Calculates macronutrient targets",
                "version": "1.0.0",
                "category": "nutrition",
                "tags": ["macros", "nutrition", "calculation"],
                "tier_requirement": "free"
            }
        }