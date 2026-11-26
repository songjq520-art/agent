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
    非本人来源连普通凭据都签不出来。
  - 无法验证来源一律 fail-closed：没凭据、凭据不是本权威签的、
    过期、越界、字段被篡改、已被撤销——执行前全部拒绝，原因码
    分清（见 REASON_*），不落任何状态。

H2 补修轮（2026-11-25，贵妃第二十五份复查 F1/F3/F4＋撤销语义）：
  - F1 签发 fail-closed 钉死：issuance_source 必须显式给出且只认
    "owner" / "system"——未标来源（None）不再是「宿主内部路径」，
    任何凭据都签不出来（旧口径把 None 释为内部路径，与 fail-closed
    自相矛盾，本轮废止）。且来源与说话人必须互相印证：主人通道
    只给主人本人签（speaker 必须是 "owner"），自称 "owner" 的
    凭据也只许从主人通道出（"system" 内部路径不许冒名签 owner）；
    自填的来源字符串单独不构成可信，组合对不上一律拒签。
  - F3 畸形一律结构化拒绝：issue / validate 对 session_id、
    scope、时间、digest、ttl 等字段先验类型与形状，畸形只返回
    拒签 / 拒绝原因码，绝不抛异常；ttl 必须为有限正数
    （NaN / inf / <= 0 拒签）。
  - F4 到期边界钉死：now >= expires_at 即失效（到期即失效）。
  - 撤销语义钉死：撤销返回后，尚未落笔的执行必须被阻断——
    validate 在给出 ok 前复查撤销状态；store 层在执行落笔前
    再复查一次（见 sqlite_store 的落笔前复查），且 store 的
    revoke_session 与受控执行共用同一把锁串行化。

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
import math
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

# 允许签发凭据的来源：主人通道与宿主内部系统路径。必须显式
# 给出——未标来源（None）一律拒签（补修 F1）；转述/第三方
# 显式来源同样拒签（与 auth_guard 的非本人判定同口径）。
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


def _is_aware_datetime(value):
    """是不是带时区的 datetime（签发只产出这种时间戳；朴素
    时间与非 datetime 一律是签发方不可能产出的形状）。"""
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.tzinfo.utcoffset(value) is not None
    )


def _credential_shape_ok(credential):
    """凭据字段形状前验（补修 F3）：字段被改成签发方不可能
    产出的类型/形状时，先于摘要计算判畸形——摘要路径对畸形
    字段（不可哈希的 id、None 时间、非 str 摘要……）会抛
    异常，验签绝不许抛。"""
    if not isinstance(credential.speaker, str) or not credential.speaker:
        return False
    if not isinstance(credential.scope, frozenset):
        return False
    if not all(isinstance(a, str) for a in credential.scope):
        return False
    if not _is_aware_datetime(credential.issued_at):
        return False
    if not _is_aware_datetime(credential.expires_at):
        return False
    if not isinstance(credential.owner_direct, bool):
        return False
    if not isinstance(credential.digest, str):
        return False
    return True


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
        ttl_seconds      有效期秒数，必须为有限正数（NaN /
                         inf / <= 0 拒签，补修 F3）；
        owner_direct     是否主人直达凭据——只许 issuance_source
                         显式为 "owner"（且 speaker 为 "owner"）
                         时签发，转述/第三方/未标来源一律拒签；
        issuance_source  签发来源：必须显式给出，只认 "owner" /
                         "system"；未标来源（None）一律拒签
                         （补修 F1：无法验证来源 fail-closed，
                         不再有「内部路径默认放行」）。来源与
                         说话人必须互相印证：主人通道只给
                         speaker="owner" 签，speaker="owner"
                         也只许从主人通道出；
        now              时钟注入（测试用），默认当前 UTC 时间；
                         注入值必须是带时区的 datetime。
        """
        if not speaker or not isinstance(speaker, str):
            return None
        # —— 签发来源 fail-closed（补修 F1）——
        if (
            not isinstance(issuance_source, str)
            or issuance_source not in _ISSUANCE_SOURCES
        ):
            # 未标来源（None）、转述/第三方显式来源（relayed、
            # third_party……）、畸形来源：一律拒签，与
            # auth_guard「非本人来源不认」的口径一致。
            return None
        if (
            issuance_source == _auth_guard.INSTRUCTION_SOURCE_OWNER
        ) != (speaker == _auth_guard.INSTRUCTION_SOURCE_OWNER):
            # 可信组合钉死：来源与说话人对不上（拿主人通道给
            # 别人签、或拿系统路径冒名签 owner）一律拒签——
            # 自填的来源字符串单独不构成可信。
            return None
        if owner_direct and (
            issuance_source != _auth_guard.INSTRUCTION_SOURCE_OWNER
        ):
            # 直达签发必须显式走主人通道（组合校验已保证此时
            # speaker 同为 "owner"）。
            return None
        try:
            scope = frozenset(scope or ())
        except TypeError:
            return None
        if not all(isinstance(a, str) for a in scope):
            return None
        try:
            allowed = frozenset(allowed_actions or ())
        except TypeError:
            return None
        if not scope or not scope.issubset(allowed):
            return None
        if isinstance(ttl_seconds, bool) or not isinstance(
            ttl_seconds, (int, float)
        ):
            return None
        try:
            ttl_value = float(ttl_seconds)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(ttl_value) or ttl_value <= 0:
            return None
        if now is not None and not _is_aware_datetime(now):
            return None
        issued_at = now or datetime.now(timezone.utc)
        try:
            expires_at = issued_at + timedelta(seconds=ttl_value)
        except OverflowError:
            return None
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
        撤销 → 字段形状（补修 F3）→ 摘要（防篡改）→ 过期 →
        范围 → 敏感动作直达要求 → 撤销复查（给出 ok 前最后一
        眼，补修撤销语义）。任何畸形输入只返回结构化拒绝，
        绝不抛异常。
        """
        verdict = {"ok": False, "reason": REASON_NO_SESSION,
                   "session_id": None}
        if not isinstance(credential, SessionCredential):
            return verdict
        if not isinstance(credential.session_id, str):
            # session_id 被改成不可哈希/非 str 的形状：连登记
            # 都查不了，按「不是本权威签的」结构化拒绝。
            return verdict
        verdict["session_id"] = credential.session_id
        if not isinstance(action, str):
            verdict["reason"] = REASON_OUT_OF_SCOPE
            return verdict
        if now is not None and not _is_aware_datetime(now):
            # 时钟注入畸形（非 datetime / 无时区）：结构化拒绝，
            # 不拿它去和凭据时间比较（比较会抛 TypeError）。
            verdict["reason"] = REASON_SESSION_TAMPERED
            return verdict
        with self._lock:
            record = self._records.get(credential.session_id)
            revoked = credential.session_id in self._revoked
        if record is None:
            return verdict
        if revoked:
            verdict["reason"] = REASON_SESSION_REVOKED
            return verdict
        if not _credential_shape_ok(credential):
            verdict["reason"] = REASON_SESSION_TAMPERED
            return verdict
        try:
            payload = _canonical_payload(
                credential.session_id, credential.speaker,
                credential.scope, credential.issued_at,
                credential.expires_at, credential.owner_direct,
            )
            digest_ok = hmac.compare_digest(
                self._digest(payload), credential.digest
            ) and hmac.compare_digest(
                record["digest"], credential.digest
            )
        except Exception:
            verdict["reason"] = REASON_SESSION_TAMPERED
            return verdict
        if not digest_ok:
            verdict["reason"] = REASON_SESSION_TAMPERED
            return verdict
        moment = now or datetime.now(timezone.utc)
        if moment >= credential.expires_at:
            # 到期即失效（补修 F4）：now == expires_at 已算过期。
            verdict["reason"] = REASON_SESSION_EXPIRED
            return verdict
        if action not in credential.scope:
            verdict["reason"] = REASON_OUT_OF_SCOPE
            return verdict
        if action in SENSITIVE_ACTIONS and not credential.owner_direct:
            verdict["reason"] = REASON_NOT_OWNER_DIRECT
            return verdict
        with self._lock:
            # 撤销复查（补修，撤销语义钉死）：验签途中凭据被
            # 撤销的，不许把这次验签放行成 ok——撤销返回后
            # 开始的判定必须看到已撤销。
            if credential.session_id in self._revoked:
                verdict["reason"] = REASON_SESSION_REVOKED
                return verdict
        verdict["ok"] = True
        verdict["reason"] = ""
        return verdict

    def is_revoked(self, session_id):
        """这张凭据当前是否已被撤销（store 落笔前复查用）。

        session_id 形状不对（非 str）按已撤销处理（fail-closed）：
        这个接口只服务于「落笔前最后一道闸」，拿不准就拦。
        """
        if not isinstance(session_id, str):
            return True
        with self._lock:
            return session_id in self._revoked

    def revoke(self, session_id):
        """撤销一张凭据（H3 接缝：凭据侧撤销登记）。

        撤销后该凭据一律 session_revoked 拒绝。返回这张凭据
        是否存在过（已撤销的重复撤销返回 False，不二次生效）。
        已执行变更的反向补偿不在这里——那是 H3 撤销接口本体。
        session_id 形状不对（非 str）返回 False，不抛异常。
        """
        if not isinstance(session_id, str):
            return False
        with self._lock:
            if session_id not in self._records:
                return False
            if session_id in self._revoked:
                return False
            self._revoked.add(session_id)
            return True
