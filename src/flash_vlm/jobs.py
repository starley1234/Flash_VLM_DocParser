"""Реестр задач конвертации для HTTP API."""

from __future__ import annotations

import asyncio
import uuid

from .schemas import Job

__all__ = ["JobManager"]


class JobManager:
    """Хранит задачи и связанные с ними фоновые asyncio-таски."""

    def __init__(self, max_jobs: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self.max_jobs = max_jobs

    def create(self, filename: str) -> Job:
        """Создаёт новую задачу и возвращает её."""
        self._trim()
        job = Job(id=uuid.uuid4().hex[:16], filename=filename)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def bind_task(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks[job_id] = task

    def remove(self, job_id: str) -> bool:
        """Удаляет задачу и отменяет её фоновый таск (если выполняется)."""
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
        return self._jobs.pop(job_id, None) is not None

    def _trim(self) -> None:
        """Удаляет старейшие завершённые задачи при превышении лимита."""
        if len(self._jobs) < self.max_jobs:
            return
        finished = [j for j in self._jobs.values() if j.status in {"done", "error"}]
        for job in finished[: max(1, len(finished) - self.max_jobs // 2)]:
            self._jobs.pop(job.id, None)
            self._tasks.pop(job.id, None)
