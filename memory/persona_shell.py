# memory/persona_shell.py（追加决定八：性情区原件在壳，2026-11-25 补建）
#
# 决定八（BOOS 2026-11-24 终审 17 条后拍板，2026-11-25 拍板
# 「17 条替换壳里旧 10 条」）：
#   - 性情区＋破甲词的原件放在十二的壳（程序基底）上——就是仓库
#     根的 SOUL.md，标记之间的那段条文；
#   - 核心记忆里那份是逐字复印件，带只读投影标记，整理员/判官/
#     降级都不许碰（落点在 sqlite_store：persona_projection 标记
#     + _is_core_protected_row + protection_tier=pinned 不占座）；
#   - 注入从壳原件生成（本模块 build_persona_block；main.py 每轮
#     拼装时从壳现读，不读记忆复印件）；
#   - 复印单向：壳→记忆。记忆侧的改动永远不许反写壳；
#   - 壳与复印件冲突：挂起问 BOOS 裁定，程序不自决、不默默覆盖
#     （校验与挂问在 sqlite_store.verify_persona_consistency）。
#
# 本模块只做纯文件/文本活：读壳、摘条文段、算哈希、拼注入块。
# 不读库、不写库、不联网；写记忆与挂问都归 sqlite_store。

import hashlib
from pathlib import Path

# 条文段在 SOUL.md 里的定界标记：逐字复印件就是这两行之间的
# 内容（首尾空白不算）。改条文只改标记之间，别动标记本身。
CLAUSES_BEGIN = "<!-- PERSONA_CLAUSES_BEGIN -->"
CLAUSES_END = "<!-- PERSONA_CLAUSES_END -->"

# 注入块上限：和回锚加强版同量级（2400 字）。17 条正文约 700
# 字，正常永远到不了顶；封顶是防壳文件被误写成长文时撑爆提示词。
PERSONA_MAX_CHARS = 2400

PERSONA_TITLE = "【性情区·壳原件】"

_PERSONA_HEADER = (
    PERSONA_TITLE
    + "以下是 BOOS 钉死的性情条文（原件在壳 SOUL.md，逐字为准）："
)


def default_shell_path():
    """壳 SOUL.md 的默认位置：仓库根（本文件在 memory/ 下）。"""
    return Path(__file__).resolve().parents[1] / "SOUL.md"


def extract_clauses_text(soul_md_text):
    """从 SOUL.md 全文里摘出标记之间的条文段（逐字，剥首尾空白）。

    标记缺失/顺序不对/段为空都返回 None——摘不出来就当没有原件，
    调用方不许拿别的内容冒充（注入宁可不带、同步宁可不印）。
    """
    if not soul_md_text:
        return None
    start = soul_md_text.find(CLAUSES_BEGIN)
    end = soul_md_text.find(CLAUSES_END)
    if start < 0 or end < 0 or end <= start:
        return None
    body = soul_md_text[start + len(CLAUSES_BEGIN):end].strip()
    return body or None


def read_shell_clauses(path=None):
    """读壳原件的条文段；读不到/摘不出返回 None（不抛）。"""
    p = Path(path) if path else default_shell_path()
    try:
        return extract_clauses_text(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def clauses_sha256(text):
    """条文段的哈希：同步/校验三方对照（壳、复印件、上次重印戳）。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_persona_block(clauses_text, max_chars=PERSONA_MAX_CHARS):
    """把壳原件条文段拼成每轮注入块。空段返回空串（不带）。

    超上限时截断加省略号（同 reanchor 第一条的口径：带一条截断
    的也比一条不带强）；正常 17 条远不到顶。
    """
    text = (clauses_text or "").strip()
    if not text:
        return ""
    block = _PERSONA_HEADER + "\n" + text
    if len(block) > max_chars:
        block = block[:max_chars] + "…"
    return block
