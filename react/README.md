# Schema Mapping Agent - A2A Compatible

A2A compatible version of the schema mapping agent that uses natural language to map between GraphQL schemas.

## Setup

1. **Environment Variables**
   Create a `.env` file with:
   ```
   WATSONX_APIKEY=your_watsonx_api_key
   WATSONX_PROJECT_ID=your_project_id
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Directory Structure**
   Make sure you have your existing directories:
   ```
   ./data/graphql/          # Your GraphQL schema files
   ./utils/                 # Your utility modules
   ./toolkit/               # Your toolkit modules
   ```

**Structure related to Schema Mapping Agent:**
```
solis-dal-ai/
└── agents/
    ├── main.py                  # FastAPI entrypoint for GraphQL Mapping Agent
    ├── schema_mapping_agent.py  # LangGraph-based GraphQL schema mapping agent logic
    ├── test_agent.py            # Script to test the agent independently
    ├── requirements.txt         # Python dependencies
    └── Dockerfile               # Docker configuration for containerized deployment
```

## Running the Agent

### Local Development
Run the commands from the root.
```bash
python agents/main.py
```

## Testing

Run the basic test:
```bash
python agents/test_agent.py
```

### Using Docker
```bash
docker build -t graphql-mapping-agent .
docker run -p 8001:8001 graphql-mapping-agent
```

## API Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /capabilities` - Agent capabilities
- `POST /execute` - Execute mapping task


## A2A Integration

This agent follows the A2A agent structure with:
- FastAPI server with standard endpoints
- Pydantic models for request/response
- Agent card metadata
- Docker containerization
- Health and capability endpoints
