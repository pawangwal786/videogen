"""Workflow artifacts router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_orchestrator, verify_auth
from app.api.schemas.artifact import ArtifactListResponse, ArtifactResponse
from app.orchestration.errors import WorkflowNotFoundError
from app.orchestration.orchestrator import WorkflowOrchestrator

router = APIRouter(prefix="/workflows", tags=["Artifacts"], dependencies=[Depends(verify_auth)])


@router.get(
    "/{workflow_id}/artifacts",
    response_model=ArtifactListResponse,
    summary="List workflow artifacts",
    description=(
        "List all artifacts produced by the specified workflow, cleanly abstracting "
        "storage provider implementation details."
    ),
    responses={
        200: {"description": "List of workflow artifacts."},
        401: {"description": "Unauthorized."},
        404: {"description": "Workflow not found."},
    },
)
async def list_artifacts_for_workflow(
    workflow_id: str,
    artifact_type: Annotated[
        str | None, Query(description="Optional filter by artifact type")
    ] = None,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)] = None,  # type: ignore[assignment]
) -> ArtifactListResponse:
    wf = await orchestrator.get_workflow(workflow_id)
    if wf is None:
        raise WorkflowNotFoundError(workflow_id)

    artifacts = await orchestrator.list_artifacts_for_workflow(
        workflow_id, artifact_type=artifact_type
    )
    artifact_responses = [ArtifactResponse.model_validate(a) for a in artifacts]
    return ArtifactListResponse(artifacts=artifact_responses, total=len(artifact_responses))
