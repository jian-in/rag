from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from .core import MemoryHub, MemoryHubSettings
from .providers import providers_from_environment


def create_hub_from_environment(
    environment: Mapping[str, str] | None = None,
) -> MemoryHub:
    env = environment if environment is not None else os.environ
    defaults = MemoryHubSettings()
    include_dirs = tuple(
        item.strip()
        for item in env.get(
            "MEMORY_VAULT_INCLUDE_DIRS", ",".join(defaults.vault_include_dirs)
        ).split(",")
        if item.strip()
    )
    settings = MemoryHubSettings(
        index_root=Path(env.get("MEMORY_HUB_INDEX_ROOT", str(defaults.index_root))),
        vault_root=Path(env.get("MEMORY_VAULT_ROOT", str(defaults.vault_root))),
        memory_root=Path(env.get("MEMORY_PROFILE_ROOT", str(defaults.memory_root))),
        rollout_root=Path(env.get("MEMORY_ROLLOUT_ROOT", str(defaults.rollout_root))),
        vault_include_dirs=include_dirs,
        chunk_size=int(env.get("MEMORY_CHUNK_SIZE", defaults.chunk_size)),
        chunk_overlap=int(env.get("MEMORY_CHUNK_OVERLAP", defaults.chunk_overlap)),
    )
    embedding, reranker = providers_from_environment(env)
    return MemoryHub(
        settings,
        embedding_provider=embedding,
        rerank_provider=reranker,
    )
