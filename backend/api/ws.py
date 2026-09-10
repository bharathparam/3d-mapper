"""WebSocket endpoint for real-time job progress streaming."""
from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.job_manager import job_manager

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """
    Subscribe to real-time progress events for a job.

    The server pushes JSON messages of type ProgressEvent.
    Connection closes when the job completes or fails.
    """
    await websocket.accept()

    job = await job_manager.get(job_id)
    if not job:
        await websocket.send_json({"event": "error", "message": f"Job {job_id} not found."})
        await websocket.close()
        return

    # Send current state immediately on connect
    await websocket.send_json({
        "event": "snapshot",
        "status": job.status,
        "progress": job.progress,
        "stages": [s.model_dump() for s in job.stages],
    })

    if job.status in ("completed", "failed"):
        await websocket.close()
        return

    await job_manager.subscribe(job_id, websocket)
    try:
        while True:
            # Keep alive — client can send pings or just wait for server events
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await job_manager.unsubscribe(job_id, websocket)
