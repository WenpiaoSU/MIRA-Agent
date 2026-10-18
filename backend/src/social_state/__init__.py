"""作用：提供 M08 版本化社会状态聚合模块。

当前功能：
- 导出可注入时钟的状态服务和过期配置。
"""

from src.social_state.service import SocialStateConfig, SocialStateService

__all__ = ["SocialStateConfig", "SocialStateService"]
