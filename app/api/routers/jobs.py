"""Workflow jobs router."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator, verify_auth
from app.api.schemas.job import JobListResponse, JobResponse
from app.orchestration.errors import WorkflowNotFoundError
from app.orchestration.orchestrator import WorkflowOrchestrator

router = APIRouter(prefix="/workflows", tags=["Jobs"], dependencies=[Depends(verify_auth)])


@router.get(
    "/{workflow_id}/jobs",
    response_model=JobListResponse,
    summary="List workflow jobs",
    description="List all pipeline stage jobs belonging to the specified workflow.",
    responses={
        200: {"description": "List of workflow jobs."},
        401: {"description": "Unauthorized."},
        404: {"description": "Workflow not found."},
    },
)
async def list_jobs_for_workflow(
    workflow_id: str,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)],
) -> JobListResponse:
    wf = await orchestrator.get_workflow(workflow_id)
    if wf is None:
        raise WorkflowNotFoundError(workflow_id)

    jobs = await orchestrator.list_jobs_for_workflow(workflow_id)
    job_responses = [JobResponse.model_validate(j) for j in jobs]
    return JobListResponse(jobs=job_responses, total=len(job_responses))
