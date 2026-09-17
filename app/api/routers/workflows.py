"""Workflow lifecycle router."""

from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse

from app.api.dependencies import (
    check_workflow_creation_rate_limit,
    get_idempotency_key,
    get_orchestrator,
    verify_auth,
)
from app.api.schemas.workflow import WorkflowCreateRequest, WorkflowResponse
from app.orchestration.errors import IdempotencyConflictError, WorkflowNotFoundError
from app.orchestration.orchestrator import WorkflowOrchestrator

router = APIRouter(prefix="/workflows", tags=["Workflows"], dependencies=[Depends(verify_auth)])


@router.post(
    "",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow",
    description=(
        "Create a new video generation workflow and asynchronously queue the initial RESEARCH job. "
        "Supports Idempotency-Key header for duplicate-safe replaying."
    ),
    dependencies=[Depends(check_workflow_creation_rate_limit)],
    responses={
        200: {"description": "Idempotent replay: existing workflow returned."},
        201: {"description": "Workflow successfully created."},
        401: {"description": "Unauthorized: missing or invalid credentials."},
        409: {"description": "Idempotency conflict: key used with different payload."},
        422: {"description": "Validation error in request payload or headers."},
        429: {"description": "Rate limit exceeded."},
    },
)
async def create_workflow(
    payload: WorkflowCreateRequest,
    idempotency_key: Annotated[str | None, Depends(get_idempotency_key)] = None,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)] = None,  # type: ignore[assignment]
) -> Response:
    if idempotency_key is not None:
        existing = await orchestrator.get_workflow_by_idempotency_key(idempotency_key)
        if existing is not None and existing.topic != payload.topic:
            raise IdempotencyConflictError(idempotency_key, existing.topic, payload.topic)

    workflow = await orchestrator.create_workflow(
        topic=payload.topic,
        idempotency_key=idempotency_key,
    )

    response_data = WorkflowResponse.model_validate(workflow).model_dump(mode="json")
    status_code = status.HTTP_201_CREATED if workflow.was_created else status.HTTP_200_OK
    return JSONResponse(status_code=status_code, content=response_data)


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponse,
    summary="Get workflow",
    description="Retrieve workflow details and current execution status by ID.",
    responses={
        200: {"description": "Workflow details."},
        401: {"description": "Unauthorized."},
        404: {"description": "Workflow not found."},
    },
)
async def get_workflow(
    workflow_id: str,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)],
) -> WorkflowResponse:
    workflow = await orchestrator.get_workflow(workflow_id)
    if workflow is None:
        raise WorkflowNotFoundError(workflow_id)
    return WorkflowResponse.model_validate(workflow)


@router.post(
    "/{workflow_id}/cancel",
    response_model=WorkflowResponse,
    summary="Cancel workflow",
    description=(
        "Fenced cancellation: locks the workflow row, transitions active jobs and attempts "
        "to CANCELLED, and rejects any subsequent worker mutations. Idempotent if already terminal."
    ),
    responses={
        200: {"description": "Workflow cancelled or already in terminal state."},
        401: {"description": "Unauthorized."},
        404: {"description": "Workflow not found."},
    },
)
async def cancel_workflow(
    workflow_id: str,
    orchestrator: Annotated[WorkflowOrchestrator, Depends(get_orchestrator)],
) -> WorkflowResponse:
    workflow = await orchestrator.cancel_workflow(workflow_id)
    return WorkflowResponse.model_validate(workflow)
