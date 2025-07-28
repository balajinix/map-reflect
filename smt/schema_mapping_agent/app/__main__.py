import logging
import os
import sys

import httpx
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from app.agent import GraphQLSchemaMappingAgent
from app.agent_executor import GraphQLSchemaMappingAgentExecutor
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def main():
    """Starts GraphQL Schema Mapping Agent server."""
    host = "localhost"
    port = 8003
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise MissingAPIKeyError("GOOGLE_API_KEY environment variable not set.")

        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        skill = AgentSkill(
            id="map_graphql_schema",
            name="GraphQL Schema Mapping",
            description="Maps and transforms GraphQL schemas between different systems and formats",
            tags=["graphql", "schema", "mapping", "transformation"],
            examples=[
                "map concert vulnerabilities to turbo market actions",
                "map concert vulnerabilities to turbo entities"
            ],
        )
        agent_card = AgentCard(
            name="GraphQL_Schema_Mapping_Agent",
            description="Maps and transforms GraphQL schemas across different systems and formats",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=GraphQLSchemaMappingAgent.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=GraphQLSchemaMappingAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=[skill],
        )

        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=GraphQLSchemaMappingAgentExecutor(),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )
        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        print(f"Starting GraphQL Schema Mapping Agent on {host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred during server startup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
