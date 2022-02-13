from __future__ import annotations

import copy
from abc import ABCMeta
from collections import OrderedDict
from typing import Iterable, Optional

import torch.nn as nn

from mmcv.runner import BaseModule as BaseMMCVModule
from mmcv.utils import Registry, build_from_cfg


class BaseModule(BaseMMCVModule, metaclass=ABCMeta):
    def __init__(
        self,
        input_key: str | Iterable[str] = "inputs",
        target_key: str | Iterable[str] = "logits",
        del_keys: str | Iterable[str] | None = None,
        init_cfg: dict | None = None,
        is_lower_trackable: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.input_key = input_key
        self.target_key = target_key
        self.del_keys = del_keys if del_keys is not None else {}

        self.is_lower_trackable = is_lower_trackable

    def __repr__(self) -> str:
        s = super().__repr__()
        s += f"\ninput_key={self.input_key}, target_key={self.target_key}, del_keys={self.del_keys}"

        return s


# TODO: add repr placeholder
# hack function to initialize any module as BaseModule without inheritance
# why?
#   Otherwise it is necessary to wrap all modules from all libraries.
#   It also allows to use any PyTorch modules (including custom ones) without inheritance
def wrap_as_base_module(
    module: nn.Module,
    init_cfg: dict | None,
    is_lower_trackable: bool,
    input_key: str | list[str] | tuple[str, ...] = "inputs",
    target_key: str | list[str] | tuple[str, ...] = "logits",
    del_keys: list[str] | tuple[str, ...] | None = None,
) -> nn.Module:
    if not hasattr(module, "input_key"):
        module.input_key = input_key
    if not hasattr(module, "target_key"):
        module.target_key = target_key
    if not hasattr(module, "del_keys"):
        module.del_keys = del_keys if del_keys is not None else {}
    if not hasattr(module, "is_lower_trackable"):
        module.is_lower_trackable = is_lower_trackable

    if isinstance(module, nn.Module):
        if not hasattr(module, "_is_init"):
            module._is_init = False
        if not hasattr(module, "is_init"):
            module.is_init = BaseModule.is_init.__get__(module)

        if not hasattr(module, "init_cfg"):
            if init_cfg is not None:
                module.init_cfg = copy.deepcopy(init_cfg)
            else:
                module.init_cfg = None

        if not hasattr(module, "init_weights"):
            module.init_weights = BaseModule.init_weights.__get__(module)
        else:
            # TODO: warning, we should replace module method for compatibility
            pass
        if not hasattr(module, "_dump_init_info"):
            module._dump_init_info = BaseModule._dump_init_info.__get__(module)
        else:
            # TODO: warning, we should replace module method for compatibility
            pass

    return module


def recursive_build(
    cfg: dict | list | tuple,
    registry: Registry,
    module_name: Optional[str] = None,
    init_cfg: dict | None = None,
    input_key: str = "inputs",
    target_key: str = "logits",
):
    if isinstance(cfg, (list, tuple)):
        cfg_dict = {f"{name}": sub_cfg for name, sub_cfg in enumerate(cfg)}
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        raise ValueError(f"Config should be dict, list or tuple, but got {type(cfg)}")

    init_cfg = cfg_dict.pop("init_cfg", init_cfg)

    input_key = cfg_dict.pop("input_key", input_key)
    target_key = cfg_dict.pop("target_key", target_key)
    del_keys = cfg_dict.pop("del_keys", ())
    module_name = cfg_dict.pop("module_name", module_name)

    is_lower_trackable: bool

    if "type" in cfg_dict:
        module = build_from_cfg(cfg=cfg_dict, registry=registry)
        is_lower_trackable = True
    else:
        module = OrderedDict()
        for idx, (name, sub_cfg) in enumerate(cfg_dict.items()):
            sub_module_name, sub_module = recursive_build(
                cfg=sub_cfg,
                module_name=f"{name}",
                registry=registry,
                init_cfg=init_cfg,
                input_key=input_key if idx == 0 else target_key,
                target_key=target_key,
            )
            module[sub_module_name] = sub_module
        module = build_from_cfg(
            cfg=dict(
                type="Pipeline",
                input_key=input_key,
                target_key=target_key,
                builded_modules=module,
            ),
            registry=registry,
        )
        is_lower_trackable = False

    if not isinstance(module, BaseModule):
        module = wrap_as_base_module(
            module=module,
            init_cfg=init_cfg,
            is_lower_trackable=is_lower_trackable,
            input_key=input_key,
            target_key=target_key,
            del_keys=del_keys,
        )

    if module_name is None:
        return module
    return (module_name, module)


build = recursive_build
build_model = build
build_pipeline = build
