from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory_hub.runtime import create_hub_from_environment


class MemoryRuntimeTests(unittest.TestCase):
    def test_runtime_uses_explicit_local_paths_without_enabling_external_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = {
                "MEMORY_HUB_INDEX_ROOT": str(root / "indexes"),
                "MEMORY_VAULT_ROOT": str(root / "vault"),
                "MEMORY_PROFILE_ROOT": str(root / "memory"),
                "MEMORY_ROLLOUT_ROOT": str(root / "rollouts"),
                "MEMORY_VAULT_INCLUDE_DIRS": "01-Projects,Daily",
                "SILICONFLOW_API_KEY": "shared-but-not-authorized",
            }

            hub = create_hub_from_environment(environment)

            self.assertEqual(hub.settings.index_root, (root / "indexes").resolve())
            self.assertEqual(hub.settings.vault_include_dirs, ("01-Projects", "Daily"))
            self.assertIsNone(hub.embedding_provider)
            self.assertIsNone(hub.rerank_provider)


if __name__ == "__main__":
    unittest.main()
