from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, Part, TextPart
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from a2a.types import InternalError, UnsupportedOperationError

from agent import GraphQLSchemaMappingAgent

class GraphQLSchemaMappingAgentExecutor(AgentExecutor):
    """A2A Executor for GraphQLSchemaMappingAgent"""

    def __init__(self):
        self.agent = GraphQLSchemaMappingAgent(base_schema_dir="./data/graphql")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Make sure the agent is initialized before execution
        await self.agent.async_init()

        # Validate incoming request
        error = self._validate_request(context)
        if error:
            raise ServerError(error=error)

        # Get user input and ensure a task exists
        user_input = context.get_user_input()
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Stream through agent
            async for step in self.agent.stream(user_input, task.contextId):
                kind = step.get("kind")
                if kind == "status-update":
                    state = step["status"]["state"]
                    message = step["status"]["message"]
                    # Extract text content
                    text = message["parts"][0]["text"]
                    final = state in (TaskState.input_required, TaskState.completed)
                    await updater.update_status(
                        state,
                        new_agent_text_message(text, task.contextId, task.id),
                        final=final,
                    )
                    if state == TaskState.input_required:
                        break
                elif kind == "artifact-update":
                    artifact = step.get("artifact")
                    text = artifact["parts"][0]["text"]
                    await updater.add_artifact(
                        [Part(root=TextPart(text=text))],
                        name="mapping_result",
                    )
                # Completed state is implied by final status-update
        except Exception as e:
            # Log or handle error
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext):
        # Return an A2A error if invalid, otherwise None
        # For now, accept any text-based message
        return None

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        # Cancellation not supported
        raise ServerError(error=UnsupportedOperationError())

