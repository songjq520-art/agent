# tools/permissions.py（第一批工具共用的权限判断）
# 作用：所有工具共用一套权限判断（GPT 的建议）——文件工具和跑命令工具
# 都调这里，不各写一套：
#   - 路径权限：固定工作区 + 真实路径比对防穿越 + 密钥黑名单 +
#     .env 只写不读（Hermes 的做法）+ 审计日志不可直接改。
#   - 命令分级：Tier 2 硬禁 / Tier 1 需确认 / Tier 0 自动放行，
#     校验在执行前跑，字符归一化 + 按 | && || ; 拆段逐段查。
#
# 诚实说明：这套分级防的是误操作和提示注入下的灾难命令，不是防存心
# 绕过的——有决心绕过总能找到办法（Hermes 官方文档原话：终端能绕过
# 文件工具的写保护）。真正的隔离靠 OS 层（Box 上先测 gVisor），
# 第一批先做到“误操作伤不到电脑”。

import re
import unicodedata
from pathlib import Path

from tools.errors import PermissionDenied, ToolError

# _SECRET_COMPONENTS：路径里任何一段出现这些名字，整个文件都不可碰。
_SECRET_COMPONENTS = {".git", ".ssh"}
# _SECRET_SUFFIXES：文件名以这些结尾，不可碰。
_SECRET_SUFFIXES = (".pem", ".key")
# _SECRET_PREFIXES：文件名以这些开头，不可碰。
_SECRET_PREFIXES = ("id_rsa",)


def _is_secret_blocked(path):
    """是不是密钥文件（黑名单）：.git/、.ssh/、*.pem、*.key、id_rsa*。"""
    for part in Path(path).parts:
        if part in _SECRET_COMPONENTS:
            return True
        lower = part.lower()
        if lower.endswith(_SECRET_SUFFIXES) or lower.startswith(_SECRET_PREFIXES):
            return True
    return False


def _is_env_file(path):
    """是不是 .env 类文件：.env、.env.local、.env.production、.envrc。"""
    for part in Path(path).parts:
        if part == ".env" or part.startswith(".env.") or part == ".envrc":
            return True
    return False


def resolve_in_workspace(workspace_root, rel_path):
    """把一个相对路径解析成工作区内的真实路径（跟随符号链接再比对）。

    越界（../ 穿越、绝对路径在工作区外、符号链接逃逸）一律拒绝。
    返回解析后的绝对路径。
    """
    root = Path(workspace_root).resolve()
    candidate = Path(rel_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionDenied(f"路径超出工作区，拒绝：{rel_path}")
    return resolved


def classify_path(workspace_root, rel_path, action):
    """判断一个文件操作允不允许：返回 (allowed, reason)。

    action 是 read / write / edit / delete / list。
    规则（按顺序查，先查到的先生效）：
      1. 超出工作区 → 拒绝。
      2. 密钥文件（.git/、.ssh/、*.pem、*.key、id_rsa*）→ 一律拒绝。
      3. .env 类文件 → 只写不读：读拒绝，写/改/删允许。
      4. .audit/ 目录 → 只读不写：审计日志不能用文件工具直接改。
      5. 其他 → 允许。
    """
    try:
        resolved = resolve_in_workspace(workspace_root, rel_path)
    except PermissionDenied as exc:
        return False, str(exc)
    if _is_secret_blocked(rel_path):
        return False, f"密钥文件，禁止{action}：{rel_path}"
    if _is_env_file(rel_path) and action == "read":
        return False, f".env 文件只写不读，禁止读取：{rel_path}"
    root = Path(workspace_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return False, f"路径超出工作区，拒绝：{rel_path}"
    if relative.parts and relative.parts[0] == ".audit" and action in ("write", "edit", "delete"):
        return False, "审计日志不可直接改写，请走审计接口"
    return True, ""


# ---------- 命令分级 ----------

# Tier 2：写死在代码里的硬禁命令（不可改配置、不可被模型改掉）。
_TIER2_EXECUTABLES = {
    "sudo", "su", "runas",  # 提权
    "wsl", "wsl.exe",  # WSL 逃逸
    "format", "shutdown", "reboot", "halt", "poweroff", "mkfs", "diskpart", "bcdedit",
}
# Tier 1：需要 BOOS 确认的命令（装东西、删东西、下载、跑自由代码）。
_SHELLS = {"sh", "bash", "zsh", "cmd", "cmd.exe", "powershell", "powershell.exe", "pwsh"}
# 下载即执行：curl/wget 的输出直接喂给 shell（整条命令上查，不按段查）。
_DOWNLOAD_EXEC_RE = re.compile(
    r"\b(curl|wget)\b[^|;&]*\|\s*(sh|bash|powershell|pwsh)\b"
    r"|\bInvoke-Expression\b|\biex\s*\(",
    re.IGNORECASE,
)
# Windows 危险组合：del /s、rd /s、format 等（按段查）。
_TIER2_SEGMENT_RES = [
    re.compile(r"^\s*rm\s+.*-[a-z]*r[a-z]*f\b.*\s(/\*|/|\*|C:\\|C:/|C:)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(rd|rmdir)\s+.*/s\b", re.IGNORECASE),
    re.compile(r"^\s*del\s+.*/s\b", re.IGNORECASE),
    re.compile(r"^\s*Remove-Item\b.*-Recurse\b.*\s(/|C:\\|C:/|C:)\s*$", re.IGNORECASE),
]


def _normalize_arg(arg):
    """字符归一化：全角转半角（防全角字符绕过检查），去首尾空白。"""
    return unicodedata.normalize("NFKC", str(arg)).strip()


def _split_segments(text):
    """按 | && || ; 拆段（引号内的不拆），逐段检查。"""
    segments = []
    current = []
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
        elif not in_single and not in_double and ch in "|&;":
            segments.append("".join(current))
            current = []
            # &&、|| 当作一个分隔符跳过。
            if i + 1 < len(text) and text[i + 1] == ch:
                i += 1
        else:
            current.append(ch)
        i += 1
    segments.append("".join(current))
    return [seg for seg in segments if seg.strip()]


def _tokens(segment):
    """一段命令按空白切词（引号内的空格不切，够检查用）。"""
    tokens = []
    current = []
    in_single = False
    in_double = False
    for ch in segment:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch.isspace() and not in_single and not in_double:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return tokens


def _check_segment(segment):
    """检查一段命令，返回 (tier, reason)，tier 是 2 / 1 / 0。"""
    tokens = _tokens(segment)
    if not tokens:
        return 0, ""
    head = tokens[0].lower()
    if head in _TIER2_EXECUTABLES:
        return 2, f"硬禁命令：{tokens[0]}"
    for pattern in _TIER2_SEGMENT_RES:
        if pattern.search(segment):
            return 2, f"硬禁命令：{segment.strip()}"
    # shell 解释器带自由代码：先按 Tier 1 处理，里面的内容另行逐段查。
    if head in _SHELLS:
        return 1, "跑 shell 自由代码，需要确认"
    if "pip" in tokens and ("install" in tokens or "uninstall" in tokens):
        return 1, "装/卸 Python 包，需要确认"
    if head == "git":
        if "push" in tokens and ("--force" in tokens or "-f" in tokens):
            return 1, "git 强制推送，需要确认"
        if "reset" in tokens and "--hard" in tokens:
            return 1, "git 硬重置，需要确认"
        if "clean" in tokens and "-f" in tokens:
            return 1, "git 强制清理，需要确认"
    if head in {"rm", "del", "rmdir", "rd", "Remove-Item", "unlink"}:
        return 1, "删除文件/目录，需要确认"
    if head in {"curl", "curl.exe", "wget", "wget.exe", "Invoke-WebRequest"}:
        return 1, "联网下载，需要确认"
    if head in {"npm", "pnpm", "yarn"} and (
        "install" in tokens or "add" in tokens or "i" in tokens
    ):
        return 1, "装 npm 包，需要确认"
    if head == "reg" and len(tokens) > 1 and tokens[1] in {"add", "delete"}:
        return 1, "改注册表，需要确认"
    return 0, ""


def classify_command(argv):
    """给一条结构化命令分级：返回 (tier, reason)。

    argv 必须是命令列表（["python", "script.py"]），不接受拼好的字符串——
    模型拼 shell 字符串是 GPT 指出的隐患，结构化调用从源头上堵住它。
    分级顺序：先查整条命令的“下载即执行”，再按段逐段查，先硬禁后需确认。
    """
    if isinstance(argv, str) or not isinstance(argv, (list, tuple)):
        raise ToolError("命令必须用列表传，比如 [\"python\", \"script.py\"]，不接受拼好的字符串")
    if not argv:
        raise ToolError("命令不能为空")
    normalized = [_normalize_arg(arg) for arg in argv]
    if not any(normalized):
        raise ToolError("命令不能为空")
    joined = " ".join(normalized)
    if _DOWNLOAD_EXEC_RE.search(joined):
        return 2, "硬禁命令：下载即执行（curl/wget 的输出喂给 shell）"
    # shell 解释器带 -c/-Command 时，把里面的自由代码也逐段查一遍。
    worst_tier = 0
    worst_reason = ""
    segments = _split_segments(joined)
    for segment in segments:
        tier, reason = _check_segment(segment)
        if tier > worst_tier:
            worst_tier, worst_reason = tier, reason
            if tier == 2:
                return 2, reason
    # sh/bash/cmd/powershell -c "..."：把引号里的内容也当命令查。
    head = normalized[0].lower()
    if head in _SHELLS and len(normalized) > 2 and normalized[1] in ("-c", "-Command", "/c"):
        inner = normalized[2]
        for segment in _split_segments(inner):
            tier, reason = _check_segment(segment)
            if tier == 2:
                return 2, reason
            if tier > worst_tier:
                worst_tier, worst_reason = tier, reason
    if worst_tier == 1:
        return 1, worst_reason
    return 0, ""
