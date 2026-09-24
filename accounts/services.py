from dataclasses import dataclass

import config
from accounts.audit import AuditLog
from accounts.auth import AuthService
from accounts.db import AppDB
from accounts.history import InsightHistory
from accounts.memory import MemoryService
from accounts.sessions import SessionTracker
from accounts.settings import AppSettings, PromptSettings


@dataclass
class Services:
    db: AppDB
    auth: AuthService
    audit: AuditLog
    history: InsightHistory
    sessions: SessionTracker
    memory: MemoryService
    prompts: PromptSettings
    settings: AppSettings


def build_services(path, llm, clock=None) -> Services:
    db = AppDB(path, clock=clock)
    history, sessions = InsightHistory(db), SessionTracker(db)
    services = Services(db, AuthService(db), AuditLog(db), history, sessions,
                        MemoryService(db, history, sessions, llm), PromptSettings(db), AppSettings(db))
    services.settings.apply(config)
    services.prompts.apply()
    return services
