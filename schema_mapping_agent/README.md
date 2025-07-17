# Schema Mapping Agent 

A2A-compatible LangGraph agent that maps between GraphQL schemas using natural language, with integrated output storage via MCP tools (Neo4j and Wiki).

---

## 🛠 Setup

### 1. Environment Variables

Create a `.env` file in the root:
WATSONX_APIKEY=your_watsonx_api_key
WATSONX_PROJECT_ID=your_project_id


### 2. Install Dependencies
```pip install -r requirements.txt```

📁 Directory Structure
```
gnn_schema_mapping/
├── app/
│   ├── __main__.py
│   ├── agent_executor.py        # A2A-compatible executor
│   ├── agent.py                 # GraphQL schema mapping agent (LangGraph)
│   ├── mcp_server.py            # Starts MCP tool server on port 3001
│   ├── neo4j_tool.py            # Tool to store mappings in Neo4j
│   ├── wiki_tool.py             # Tool to update schema wiki
│   ├── test_agent.py            # Test A2A connection to MCP tools
│   └── utils/                   # Utility functions
...
```

## Running the Agent
Step-by-Step:

Start MCP Tool Server: ```python app/mcp_server.py```   # Loads neo4j_tool and wiki_tool on port 3001


Start Agent: ```python app/agent.py```      # LangGraph agent available on port 8001


Testing: Run test with A2A tool call: ```python app/test_agent.py```


## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /capabilities` - Agent capabilities
- `POST /execute` - Execute mapping task


## Characteristics

This agent follows the A2A agent structure with:
- FastAPI server with standard endpoints
- Dynamic tool loading via MCP server (mcp_server.py)
- Agent card metadata
- Intent-based tool routing (populate_neo4j, populate_wiki)
- Health and capability endpoints
