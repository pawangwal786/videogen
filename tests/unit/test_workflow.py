from app.orchestrator.state import TaskStatus, WorkflowStatus
from app.orchestrator.task import Task
from app.orchestrator.workflow import Workflow


def test_workflow_defaults_to_created():
    workflow = Workflow(topic="AI infrastructure")

    assert workflow.status == WorkflowStatus.CREATED
    assert workflow.topic == "AI infrastructure"


def test_task_defaults_to_pending():
    workflow = Workflow(topic="AI infrastructure")

    task = Task(
        workflow_id=workflow.id,
        agent_name="research",
    )

    assert task.status == TaskStatus.PENDING
    assert task.attempt == 0
