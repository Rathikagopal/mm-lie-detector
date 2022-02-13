from __future__ import annotations

from ..registry import registry
from .timesformer_head import TimesformerHead


@registry.register_module()
class ASTHead(TimesformerHead):
    pass
