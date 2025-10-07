#!/usr/bin/env python3
"""
Custom Tools Example

This example demonstrates how to create and register custom tools
with the RAGAgent.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, BaseTool, ToolParameter, ToolStatus
from pydantic import Field


class WeatherTool(BaseTool):
    """Custom tool for getting weather information."""
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get current weather information for a location"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="location",
                type="string",
                description="City name or location",
                required=True
            ),
            ToolParameter(
                name="units",
                type="string",
                description="Temperature units (celsius or fahrenheit)",
                required=False,
                default="celsius",
                enum=["celsius", "fahrenheit"]
            )
        ]
    
    async def _execute(self, location: str, units: str = "celsius", **kwargs) -> Dict[str, Any]:
        """Execute the weather tool."""
        # Simulate weather API call
        temperature = 22 if units == "celsius" else 72
        
        return {
            "location": location,
            "temperature": temperature,
            "units": units,
            "condition": "Sunny",
            "humidity": 65,
            "wind_speed": 10
        }


class CalculatorTool(BaseTool):
    """Custom tool for performing calculations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Perform mathematical calculations"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def _setup_parameters(self) -> None:
        """Set up tool parameters."""
        self._parameters = [
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True
            )
        ]
    
    async def _execute(self, expression: str, **kwargs) -> Dict[str, Any]:
        """Execute the calculator tool."""
        try:
            # Simple evaluation (in production, use a safe math parser)
            result = eval(expression, {"__builtins__": {}}, {})
            return {
                "expression": expression,
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e),
                "success": False
            }


async def main():
    """Custom tools example."""
    print("=" * 70)
    print("Custom Tools Example")
    print("=" * 70)
    
    # Initialize agent
    print("\n1. Initializing RAGAgent...")
    agent = RAGAgent()
    print(f"   Initial tools: {len(agent.get_available_tools())}")
    
    # Create custom tools
    print("\n2. Creating custom tools...")
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    print(f"   Created: {weather_tool.name}")
    print(f"   Created: {calculator_tool.name}")
    
    # Register custom tools
    print("\n3. Registering custom tools...")
    agent.add_tool(weather_tool, category="utility")
    agent.add_tool(calculator_tool, category="utility")
    print(f"   Total tools: {len(agent.get_available_tools())}")
    
    # List all tools
    print("\n4. Available tools:")
    for tool_name in agent.get_available_tools():
        info = agent.get_tool_info(tool_name)
        print(f"   - {tool_name}: {info['description']}")
    
    # Use custom weather tool
    print("\n5. Using weather tool...")
    weather_result = await weather_tool.run(
        location="San Francisco",
        units="celsius"
    )
    print(f"   Location: {weather_result.data['location']}")
    print(f"   Temperature: {weather_result.data['temperature']}°{weather_result.data['units'][0].upper()}")
    print(f"   Condition: {weather_result.data['condition']}")
    
    # Use custom calculator tool
    print("\n6. Using calculator tool...")
    calc_result = await calculator_tool.run(
        expression="2 + 2 * 3"
    )
    print(f"   Expression: {calc_result.data['expression']}")
    print(f"   Result: {calc_result.data['result']}")
    
    # Get tool information
    print("\n7. Tool information:")
    weather_info = agent.get_tool_info("get_weather")
    print(f"   Tool: {weather_info['name']}")
    print(f"   Version: {weather_info['version']}")
    print(f"   Parameters: {list(weather_info['parameters'].keys())}")
    print(f"   Required: {weather_info['required_parameters']}")
    
    print("\n" + "=" * 70)
    print("Custom tools example complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- Custom tools extend BaseTool")
    print("- Define parameters with ToolParameter")
    print("- Implement _execute() method")
    print("- Register with agent.add_tool()")
    print("- Tools are automatically validated")


if __name__ == "__main__":
    asyncio.run(main())
