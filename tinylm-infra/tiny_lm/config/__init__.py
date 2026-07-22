# tiny_lm/config/__init__.py

"""
TinyLM 项目的配置管理模块。

外部代码应优先从 tiny_lm.config 导入配置接口，
而不是直接依赖 loader.py、schema.py 等内部文件。
"""

from tiny_lm.config.loader import load_config
from tiny_lm.config.schema import Config
from tiny_lm.config.validation import validate_config


# 限定使用 `from tiny_lm.config import *` 时导出的名称，
# 同时也说明该模块对外提供的稳定接口。
__all__ = [
    "Config",
    "load_config",
    "validate_config",
]