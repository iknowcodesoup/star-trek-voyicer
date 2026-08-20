"""FastAPI dependency providers -- Provider pattern.

Consolidates every route's Depends() source in one place, matching janewav's
own src/dependencies.py precedent.
"""

from fastapi import Request

from jeanlucrecord.core.job_runner import JobRunner
from jeanlucrecord.infrastructure.webhook_gateway import WebhookNotifier


def get_job_runner(request: Request) -> JobRunner:
    return request.app.state.job_runner


def get_webhook_notifier(request: Request) -> WebhookNotifier:
    return request.app.state.webhook_notifier
