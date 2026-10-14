"""作用：验证服务端签发的浏览器身份凭据。

当前功能：
- 使用 HttpOnly Cookie 关联长期用户，不接受客户端指定 user_id。
- 校验会话拥有者，避免仅凭 session_id 读取其他浏览器的记忆。
"""

import secrets

from src.repositories.memory import token_digest


COOKIE_NAME = "mira_identity"


def owns_session(connection, session) -> bool:
    """匿名会话保持原行为；有长期身份的会话必须提供匹配凭据。"""
    if not session.owner_token_hash:
        return True
    token = getattr(connection, "cookies", {}).get(COOKIE_NAME, "")
    return bool(token and len(token) <= 128 and secrets.compare_digest(token_digest(token), session.owner_token_hash))
