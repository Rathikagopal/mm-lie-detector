from __future__ import annotations

from ..registry import registry
from .timesformer_head import TimeSformerHead


@registry.register_module()
class ASTHead(TimeSformerHead):
    pass
