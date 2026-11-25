# -*- coding: utf-8 -*-
"""会话认证 / 可信直达授权范围（四件套·立项 H2，追加决定十六，2026-11-25）。

要解决什么（施工清单 H2）：问题一的授权硬判（auth_guard）已能把
「转述来的授权」判成未核实声称，但宿主一直没有「已验证会话」这个
载体——受控入口（立项 H1 的 submit_controlled_proposal）执行改记忆
类动作时，除了 instruction_source 这个调用方自填的字符串，没有
任何东西能证明「这条指令确实来自 BOOS 本人在一个已验证会话里」。
本模块补这一层：宿主签发带范围的会话凭据，H1 入口在生产执行开关
打开后、真正执行前强制验凭据。

口径（契约三＋BOOS 定版）：
  - 授权类断言只认 BOOS 本人在已验证会话里亲口所说；受控入口执行
    改记忆类动作只认本人指令（程序硬判，不依赖模型当场判断）。
  - 「可信直达」范围要窄：敏感动作（本轮定为保护变更
    set_protection_tier——契约三明列「保护变更只认 BOOS 本人指令」）
    只许主人直达会话（owner_direct）授权；普通会话凭据范围再全
    也碰不了敏感动作。
  - 转述/第三方来源永远拿不到直达凭据：签发直达凭据必须显式走
    主人通道（issuance_source == "owner"），auth_guard 口径下
    非本人来源连普通凭据都签不出来。签发是新接口、没有老调用方，
    所以这里不沿用 is_owner_instruction 对 None 的放行——直达
    签发必须显式标 "owner"，None 不算。
  - 无法验证来源一律 fail-closed：没凭据、凭据不是本权威签的、
    过期、越界、字段被篡改、已被撤销——执行前全部拒绝，原因码
    分清（见 REASON_*），不落任何状态。

载体与边界（诚实说明）：
  - 凭据是内存对象：签发登记与签名密钥只活在签发权威的进程内存
    里，不落盘、不进账本。进程重启即全部失效，须重新签发——会话
    本来就是短命的，这个粒度够用；跨进程/跨重启的会话不在本轮。
  - web 层现状没有会话/说话人概念（web/server.py 全文无 session
    概念），本轮不动 web 层：接缝以 store 层凭据校验为准，将来
    web 层接入会话时，只许它把会话凭据透传进提案，不许它自签。
  - 签发只供宿主内部流程与测试调用，不存在默认后门凭据：权威
    构造时登记为空，谁要凭据谁显式签发，签发条件如上硬判。
  - 撤销（H3）接缝：revoke_session 先做凭据侧撤销登记（被撤凭据
    此后一律拒）；已执行变更的反向补偿是 H3 的撤销接口本体，
    不在本模块。
"""

import hashlib
import hmac
import json
import secrets
import threading
from datetime import datetime, timedelta, timezone

from memory import auth_guard as _auth_guard

# 验凭据的原因码（回执 reason 直接用这些值，调用方按码分流）：
REASON_NO_SESSION = "no_session"                # 没带凭据 / 不是本权威签的
REASON_SESSION_EXPIRED = "session_expired"      # 凭据已过期
REASON_OUT_OF_SCOPE = "out_of_scope"            # 动作不在凭据范围里
REASON_NOT_OWNER_DIRECT = "not_owner_direct"    # 敏感动作但不是主人直达会话
REASON_SESSION_TAMPERED = "session_tampered"    # 凭据字段被改过（摘要对不上）
REASON_SESSION_REVOKED = "session_revoked"      # 凭据已被撤销（H3 接缝）

# 敏感动作集：只许主人直达会话授权。保护变更是契约三明列的
# 「只认 BOOS 本人指令」一类；其余受控动作（降级/改抽屉/标记
# 休眠）是整理员的常规动作，凭据在范围内即可。
SENSITIVE_ACTIONS = frozenset({"set_protection_tier"})

# 允许签发凭据的来源：主人通道与宿主内部系统路径。转述/第三方
# 显式来源一律拒签（与 auth_guard 的非本人判定同口径）。
_ISSUANCE_SOURCES = frozenset({_auth_guard.INSTRUCTION_SOURCE_OWNER, "system"})


def _canonical_payload(session_id, speaker, scope, issued_at,
                       expires_at, owner_direct):
    """凭据字段的规范化载荷（摘要的输入；字段顺序无关）。"""
    return json.dumps(
        {
            "session_id": session_id,
            "speaker": speaker,
            "scope": sorted(scope),
            "issued_at": issued_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "owner_direct": bool(owner_direct),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


class SessionCredential:
    """一张会话凭据（内存对象）。

    字段公开可读，但任何改动都会让摘要对不上、验凭据时按
    session_tampered 拒绝——要改范围/期限只能重新签发。
    """

    __slots__ = (
        "session_id", "speaker", "scope", "issued_at",
        "expires_at", "owner_direct", "digest",
    )

    def __init__(self, session_id, speaker, scope, issued_at,
                 expires_at, owner_direct, digest):
        self.session_id = session_id
        self.speaker = speaker
        self.scope = frozenset(scope)
        self.issued_at = issued_at
        self.expires_at = expires_at
        self.owner_direct = bool(owner_direct)
        self.digest = digest

    def __repr__(self):
        return (
            "SessionCredential(id=%s, speaker=%r, scope=%s, "
            "owner_direct=%s, expires=%s)"
            % (
                self.session_id, self.speaker, sorted(self.scope),
                self.owner_direct, self.expires_at.isoformat(),
            )
        )


class SessionAuthority:
    """会话凭据的签发权威（进程级，登记与密钥只在内存）。

    一个 store 实例持一个权威；凭据只能在本权威验，换个实例
    （哪怕同目录重开）一律 no_session——跨进程会话不在本轮范围。
    """

    def __init__(self):
        self._secret = secrets.token_bytes(32)
        self._lock = threading.Lock()
        self._records = {}   # session_id -> {"payload": str, "digest": str}
        self._revoked = set()

    def _digest(self, payload):
        return hmac.new(
            self._secret, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def issue(self, *, speaker, scope, ttl_seconds=3600,
              owner_direct=False, issuance_source=None,
              allowed_actions=None, now=None):
        """签发一张会话凭据；口径不合返回 None（不抛、不留痕外泄）。

        speaker          说话人/来源身份（如 "owner"、"curator"）；
        scope            授权范围（可执行的动作集），必须是
                         allowed_actions 的子集（动作全集由受控
                         入口定义，本模块不另立第二份口径）；
        ttl_seconds      有效期秒数，必须为正；
        owner_direct     是否主人直达凭据——只许 issuance_source
                         显式为 "owner" 时签发，转述/第三方/未标
                         来源一律拒签直达；
        issuance_source  签发来源：只认 "owner" / "system" / None
                         （宿主内部路径）；其余显式来源（relayed、
                         third_party 等）连普通凭据都拒签；
        now              时钟注入（测试用），默认当前 UTC 时间。
        """
        if not speaker or not isinstance(speaker, str):
            return None
        try:
            scope = frozenset(scope or ())
        except TypeError:
            return None
        allowed = frozenset(allowed_actions or ())
        if not scope or not scope.issubset(allowed):
            return None
        if not isinstance(ttl_seconds, (int, float)) or ttl_seconds <= 0:
            return None
        if (
            issuance_source is not None
            and issuance_source not in _ISSUANCE_SOURCES
        ):
            # 转述/第三方显式来源（relayed、third_party……）拒签：
            # 与 auth_guard「非本人来源不认」的口径一致。
            return None
        if owner_direct and (
            issuance_source != _auth_guard.INSTRUCTION_SOURCE_OWNER
        ):
            # 直达签发必须显式走主人通道：None（未标来源）不算，
            # 这是新接口、没有老调用方的兼容包袱。
            return None
        issued_at = now or datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(seconds=float(ttl_seconds))
        session_id = secrets.token_hex(16)
        payload = _canonical_payload(
            session_id, speaker, scope, issued_at, expires_at,
            owner_direct,
        )
        digest = self._digest(payload)
        with self._lock:
            self._records[session_id] = {
                "payload": payload, "digest": digest,
            }
        return SessionCredential(
            session_id, speaker, scope, issued_at, expires_at,
            owner_direct, digest,
        )

    def validate(self, credential, action, *, now=None):
        """验一张凭据能不能授权某个动作。

        返回 {"ok", "reason", "session_id"}：ok 时 reason 为空串；
        不 ok 时 reason 是 REASON_* 之一。判定顺序：存在性 →
        撤销 → 摘要（防篡改）→ 过期 → 范围 → 敏感动作直达要求。
        """
        verdict = {"ok": False, "reason": REASON_NO_SESSION,
                   "session_id": None}
        if not isinstance(credential, SessionCredential):
            return verdict
        verdict["session_id"] = credential.session_id
        with self._lock:
            record = self._records.get(credential.session_id)
            revoked = credential.session_id in self._revoked
        if record is None:
            return verdict
        if revoked:
            verdict["reason"] = REASON_SESSION_REVOKED
            return verdict
        payload = _canonical_payload(
            credential.session_id, credential.speaker,
            credential.scope, credential.issued_at,
            credential.expires_at, credential.owner_direct,
        )
        if (
            not hmac.compare_digest(self._digest(payload),
                                    credential.digest or "")
            or not hmac.compare_digest(record["digest"],
                                       credential.digest or "")
        ):
            verdict["reason"] = REASON_SESSION_TAMPERED
            return verdict
        moment = now or datetime.now(timezone.utc)
        if moment > credential.expires_at:
            verdict["reason"] = REASON_SESSION_EXPIRED
            return verdict
        if action not in credential.scope:
            verdict["reason"] = REASON_OUT_OF_SCOPE
            return verdict
        if action in SENSITIVE_ACTIONS and not credential.owner_direct:
            verdict["reason"] = REASON_NOT_OWNER_DIRECT
            return verdict
        verdict["ok"] = True
        verdict["reason"] = ""
        return verdict

    def revoke(self, session_id):
        """撤销一张凭据（H3 接缝：凭据侧撤销登记）。

        撤销后该凭据一律 session_revoked 拒绝。返回这张凭据
        是否存在过（已撤销的重复撤销返回 False，不二次生效）。
        已执行变更的反向补偿不在这里——那是 H3 撤销接口本体。
        """
        with self._lock:
            if session_id not in self._records:
                return False
            if session_id in self._revoked:
                return False
            self._revoked.add(session_id)
            return True
