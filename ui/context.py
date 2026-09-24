from dataclasses import dataclass


@dataclass
class Ctx:
    store: object
    ingest_node: object
    graph: object
    llms: tuple
    services: object
    user: dict            # {"id", "username", "role"}
    session_id: str
    shared: dict
    messages: list
