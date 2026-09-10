"""Synthetic DeepSeek capabilities shared by offline tests; no credentials."""
from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock

from freetokenapi.deepseek.models import parse_default_model

DEFAULT_MODEL = {
    "model_type": "default", "name": "Unified", "enabled": True, "switchable": True, "is_default": True,
    "think_feature": {}, "search_feature": {},
    "file_feature": {"vision": True, "conflict_with_search": False, "max_input_file_count": 50, "max_upload_file_size": 104857600},
}


def model_configs(**changes):
    model = copy.deepcopy(DEFAULT_MODEL)
    model.update(changes)
    return [model]


def capable_account_mock():
    account = MagicMock(index=0, broken=False)
    account.model_config.get = AsyncMock(return_value=parse_default_model(model_configs()))
    return account
