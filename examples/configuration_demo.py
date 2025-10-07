#!/usr/bin/env python3
"""
Configuration Management Demo

This example demonstrates all the different ways to configure the RAGAgent:
1. From YAML file
2. From JSON file
3. From environment variables
4. Programmatically with config objects
5. Saving and loading configurations
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lexora import RAGAgent, LLMConfig, VectorDBConfig, AgentConfig, RAGAgentConfig


def demo_yaml_config():
    """Demonstrate loading configuration from YAML file."""
    print("=" * 70)
    print("1. Loading Configuration from YAML File")
    print("=" * 70)
    
    try:
        # Load agent from YAML configuration
        agent = RAGAgent.from_yaml("examples/config_example.yaml")
        print(f"✓ Agent loaded from YAML: {agent}")
        print(f"  - LLM Provider: {agent.llm_config.provider}")
        print(f"  - Model: {agent.llm_config.model}")
        print(f"  - Vector DB: {agent.vector_db_config.provider}")
        print(f"  - Tools: {len(agent.get_available_tools())}")
    except Exception as e:
        print(f"✗ Failed to load from YAML: {e}")


def demo_json_config():
    """Demonstrate loading configuration from JSON file."""
    print("\n" + "=" * 70)
    print("2. Loading Configuration from JSON File")
    print("=" * 70)
    
    try:
        # Load agent from JSON configuration
        agent = RAGAgent.from_json("examples/config_example.json")
        print(f"✓ Agent loaded from JSON: {agent}")
        print(f"  - LLM Provider: {agent.llm_config.provider}")
        print(f"  - Model: {agent.llm_config.model}")
        print(f"  - Vector DB: {agent.vector_db_config.provider}")
        print(f"  - Tools: {len(agent.get_available_tools())}")
    except Exception as e:
        print(f"✗ Failed to load from JSON: {e}")


def demo_env_config():
    """Demonstrate loading configuration from environment variables."""
    print("\n" + "=" * 70)
    print("3. Loading Configuration from Environment Variables")
    print("=" * 70)
    
    # Set example environment variables
    os.environ["LEXORA_LLM_MODEL"] = "gpt-3.5-turbo"
    os.environ["LEXORA_LLM_PROVIDER"] = "openai"
    os.environ["LEXORA_VECTORDB_PROVIDER"] = "faiss"
    os.environ["LEXORA_VECTORDB_CONNECTION_PARAMS"] = '{"index_path": "./faiss_index"}'
    
    try:
        # Load agent from environment variables
        agent = RAGAgent.from_env()
        print(f"✓ Agent loaded from environment: {agent}")
        print(f"  - LLM Provider: {agent.llm_config.provider}")
        print(f"  - Model: {agent.llm_config.model}")
        print(f"  - Vector DB: {agent.vector_db_config.provider}")
        print(f"  - Tools: {len(agent.get_available_tools())}")
    except Exception as e:
        print(f"✗ Failed to load from environment: {e}")


def demo_programmatic_config():
    """Demonstrate programmatic configuration."""
    print("\n" + "=" * 70)
    print("4. Programmatic Configuration")
    print("=" * 70)
    
    # Create configuration objects
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY", "test-key"),
        temperature=0.7,
        max_tokens=2000
    )
    
    vector_db_config = VectorDBConfig(
        provider="faiss",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Mock for testing
        dimension=384,
        connection_params={
            "index_path": "./faiss_index",
            "metric": "cosine"
        }
    )
    
    agent_config = AgentConfig(
        max_context_length=8000,
        max_tool_calls=10,
        log_level="INFO"
    )
    
    # Create agent with config objects
    agent = RAGAgent(
        llm_config=llm_config,
        vector_db_config=vector_db_config,
        agent_config=agent_config
    )
    
    print(f"✓ Agent created programmatically: {agent}")
    print(f"  - LLM Provider: {agent.llm_config.provider}")
    print(f"  - Model: {agent.llm_config.model}")
    print(f"  - Vector DB: {agent.vector_db_config.provider}")
    print(f"  - Tools: {len(agent.get_available_tools())}")


def demo_config_from_dict():
    """Demonstrate loading from dictionary."""
    print("\n" + "=" * 70)
    print("5. Loading Configuration from Dictionary")
    print("=" * 70)
    
    config_dict = {
        "llm": {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "vector_db": {
            "provider": "faiss",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "connection_params": {
                "index_path": "./faiss_index"
            }
        },
        "agent": {
            "max_context_length": 8000,
            "max_tool_calls": 10,
            "log_level": "INFO"
        }
    }
    
    # Create config from dictionary
    config = RAGAgentConfig.from_dict(config_dict)
    agent = RAGAgent.from_config(config)
    
    print(f"✓ Agent created from dictionary: {agent}")
    print(f"  - LLM Provider: {agent.llm_config.provider}")
    print(f"  - Model: {agent.llm_config.model}")
    print(f"  - Vector DB: {agent.vector_db_config.provider}")
    print(f"  - Tools: {len(agent.get_available_tools())}")


def demo_save_config():
    """Demonstrate saving configuration to files."""
    print("\n" + "=" * 70)
    print("6. Saving Configuration to Files")
    print("=" * 70)
    
    # Create an agent
    agent = RAGAgent()
    
    # Save configuration to YAML
    yaml_path = "examples/saved_config.yaml"
    agent.save_config(yaml_path, format="yaml")
    print(f"✓ Configuration saved to {yaml_path}")
    
    # Save configuration to JSON
    json_path = "examples/saved_config.json"
    agent.save_config(json_path, format="json")
    print(f"✓ Configuration saved to {json_path}")
    
    # Get current config
    current_config = agent.get_config()
    print(f"✓ Current config retrieved: {type(current_config).__name__}")


def demo_individual_config_from_env():
    """Demonstrate loading individual configs from environment."""
    print("\n" + "=" * 70)
    print("7. Loading Individual Configs from Environment")
    print("=" * 70)
    
    # Set environment variables
    os.environ["LEXORA_LLM_MODEL"] = "gpt-4"
    os.environ["LEXORA_LLM_TEMPERATURE"] = "0.5"
    
    try:
        # Load individual config from environment
        llm_config = LLMConfig.from_env()
        print(f"✓ LLM Config from env:")
        print(f"  - Provider: {llm_config.provider}")
        print(f"  - Model: {llm_config.model}")
        print(f"  - Temperature: {llm_config.temperature}")
    except Exception as e:
        print(f"✗ Failed to load LLM config: {e}")
    
    try:
        agent_config = AgentConfig.from_env()
        print(f"✓ Agent Config from env:")
        print(f"  - Max context: {agent_config.max_context_length}")
        print(f"  - Max tool calls: {agent_config.max_tool_calls}")
        print(f"  - Log level: {agent_config.log_level}")
    except Exception as e:
        print(f"✗ Failed to load agent config: {e}")


def print_configuration_guide():
    """Print a guide for configuration options."""
    print("\n" + "=" * 70)
    print("Configuration Guide")
    print("=" * 70)
    
    guide = """
Configuration Methods:

1. YAML File:
   agent = RAGAgent.from_yaml("config.yaml")

2. JSON File:
   agent = RAGAgent.from_json("config.json")

3. Environment Variables:
   agent = RAGAgent.from_env()

4. Programmatic:
   agent = RAGAgent(
       llm_config=LLMConfig(...),
       vector_db_config=VectorDBConfig(...),
       agent_config=AgentConfig(...)
   )

5. From Dictionary:
   config = RAGAgentConfig.from_dict(config_dict)
   agent = RAGAgent.from_config(config)

Saving Configuration:
   agent.save_config("config.yaml", format="yaml")
   agent.save_config("config.json", format="json")

Environment Variable Prefixes:
   - LEXORA_LLM_*       : LLM configuration
   - LEXORA_VECTORDB_*  : Vector DB configuration
   - LEXORA_AGENT_*     : Agent configuration

See examples/.env.example for all available environment variables.
"""
    print(guide)


def main():
    """Run all configuration demos."""
    print("\n" + "=" * 70)
    print("RAGAgent Configuration Management Demo")
    print("=" * 70)
    
    # Run all demos
    demo_yaml_config()
    demo_json_config()
    demo_env_config()
    demo_programmatic_config()
    demo_config_from_dict()
    demo_save_config()
    demo_individual_config_from_env()
    
    # Print guide
    print_configuration_guide()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
