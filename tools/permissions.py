# tools/permissions.py（第一批工具共用的权限判断）
# 作用：所有工具共用一套权限判断（GPT 的建议）——文件工具和跑命令工具
# 都调这里，不各写一套：
#   - 路径权限：固定工作区 + 真实路径比对防穿越 + 密钥黑名单 +
#     .env 只写不读（Hermes 的做法）+ 审计日志不可直接改。
#     密钥/.env 的判断看解析后的真实路径，不只看用户输入的名字；
#     .backups/ 下的备份按它备份的原文件判断。
#   - 命令分级：Tier 2 硬禁 / Tier 1 需确认 / Tier 0 自动放行，
#     校验在执行前跑，字符归一化 + 按 | && || ; 拆段逐段查；
#     命令头先归一化（去路径、去 .exe/.com/.bat/.cmd 后缀、小写）再比对，
#     全路径和带后缀的写法绕不过硬禁；解释器内联代码
#     （python -c / -m、node -e 等）是 Tier 1。
#   - 路径保护比对一律大小写不敏感：Windows 文件系统不分大小写，
#     .ENV 就是 .env，.GIT 就是 .git，大写写法绕不过保护。
#   - 确认有效性：确认必须当场、当次、来自 BOOS 本人；带作用域的确认
#     必须和本次操作完全一致，一次确认只顶一次操作。
#
# 诚实说明：这套分级防的是误操作和提示注入下的灾难命令，不是防存心
# 绕过的——有决心绕过总能找到办法（Hermes 官方文档原话：终端能绕过
# 文件工具的写保护）。真正的隔离靠 OS 层（Box 上先测 gVisor），
# 第一批先做到“误操作伤不到电脑”。

import base64
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
    """是不是密钥文件（黑名单）：.git/、.ssh/、*.pem、*.key、id_rsa*。
    比对大小写不敏感：Windows 上 .GIT 就是 .git。"""
    for part in Path(path).parts:
        if part.lower() in _SECRET_COMPONENTS:
            return True
        lower = part.lower()
        if lower.endswith(_SECRET_SUFFIXES) or lower.startswith(_SECRET_PREFIXES):
            return True
    return False


def _is_env_file(path):
    """是不是 .env 类文件：.env、.env.local、.env.production、.envrc。
    比对大小写不敏感：Windows 上 .ENV 就是 .env。"""
    for part in Path(path).parts:
        low = part.lower()
        if low == ".env" or low.startswith(".env.") or low == ".envrc":
            return True
    return False


# 备份文件名：<原路径编码后>_<时间戳>.bak，时间戳是 %Y%m%dT%H%M%S%f。
# 路径分隔符用 %2F 式转义（% → %25，/ → %2F，\ → %5C）：以前
# 是把 / 换成 __，a/b.txt 和字面就叫 a__b.txt 的文件会撞同一个
# 备份名；% 编码是一一对应的，撞不上。_original_from_backup_name
# 解码时认新编码；没有 %2F/%5C/%25 的旧备份按老办法（__ → /）解，
# 老备份不至于一下子认不出来。
_BACKUP_NAME_RE = re.compile(r"^(.*)_(\d{8}T\d{12})\.bak$")


def _encode_backup_name(rel_path):
    """把工作区里的相对路径编成备份文件名里的一段：/ 和 \ 用 %2F/%5C
    转义，% 先转成 %25（不先转 %，字面带 %2F 的文件名解码时会串）。"""
    return (
        str(rel_path)
        .replace("%", "%25")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )


def _original_from_backup_name(filename):
    """把 .backups/ 里的备份文件名还原成它备份的原路径（工作区里的相对路径）。
    还原不出来（不是我们生成的备份名）返回 None。"""
    match = _BACKUP_NAME_RE.match(filename)
    if not match:
        return None
    encoded = match.group(1)
    if "%2F" in encoded or "%5C" in encoded or "%25" in encoded:
        return (
            encoded.replace("%2F", "/")
            .replace("%5C", "\\")
            .replace("%25", "%")
        )
    return encoded.replace("__", "/")


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
    密钥/.env 的判断看解析后的真实路径（工作区里的相对部分），
    不只看用户输入的名字——不然工作区里放一个指到 .env 的符号链接，
    读链接名就把 .env 读出来了。备份文件（.backups/ 下的 .bak）
    按它备份的原文件来判断，免得 .env 的备份成了漏密钥的口子。
    """
    try:
        resolved = resolve_in_workspace(workspace_root, rel_path)
    except PermissionDenied as exc:
        return False, str(exc)
    root = Path(workspace_root).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return False, f"路径超出工作区，拒绝：{rel_path}"
    check_rel = relative
    if relative.parts and relative.parts[0].lower() == ".backups" and len(relative.parts) == 2:
        original = _original_from_backup_name(relative.parts[1])
        if original is not None:
            check_rel = Path(original)
    if _is_secret_blocked(check_rel):
        return False, f"密钥文件，禁止{action}：{rel_path}"
    if _is_env_file(check_rel) and action == "read":
        return False, f".env 文件只写不读，禁止读取：{rel_path}"
    if relative.parts and relative.parts[0].lower() == ".audit" and action in ("write", "edit", "delete"):
        return False, "审计日志不可直接改写，请走审计接口"
    return True, ""


# ---------- 命令分级 ----------

# Tier 2：写死在代码里的硬禁命令（不可改配置、不可被模型改掉）。
# 只写基本名：比对前命令头会先归一化（去路径、去后缀、小写），
# 所以 shutdown.exe、完整路径、大小写混写都落到同一个名字上。
_TIER2_EXECUTABLES = {
    "sudo", "su", "runas",  # 提权
    "wsl",  # WSL 逃逸
    "format", "shutdown", "reboot", "halt", "poweroff", "mkfs", "diskpart", "bcdedit",
}
# Tier 1：需要 BOOS 确认的命令（装东西、删东西、下载、跑自由代码）。
# 同样只写基本名，比对走归一化后的名字。
_SHELLS = {"sh", "bash", "zsh", "cmd", "powershell", "pwsh"}
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

# 内联代码解释器：python -c / python -m、node -e 这类“现拼的代码”
# 不经过文件，直接跑——必须当 Tier 1（要 BOOS 确认）。
# 注意：python script.py（跑已经写好的脚本文件）还是 Tier 0，
# 因为文件写进去的时候已经要过一次确认；内联代码没经过这一关。
# 完整路径也认（比如 .venv/bin/python）：只看可执行文件的基本名。
_INTERPRETER_INLINE_FLAGS = {
    "python": {"-c", "-m", "-"},
    "python3": {"-c", "-m", "-"},
    "py": {"-c", "-m", "-"},
    "node": {"-e", "--eval"},
    "ruby": {"-e"},
    "perl": {"-e"},
    "php": {"-r"},
}


def _program_name(token):
    """取命令头的可执行文件名（去路径、去后缀、小写），比如
    "C:/x/.venv/Scripts/python.exe" → "python"，
    "C:\\Windows\\System32\\shutdown.exe" → "shutdown"。
    反斜杠先换成正斜杠再取最后一段：在 Linux 上跑 Windows 路径的
    分类测试时，pathlib 会把反斜杠当普通字符，直接取 name 会拿到
    整条路径，所以这里不用 Path。Windows 上可执行的后缀不只 .exe，
    .com、.bat、.cmd 也去掉。"""
    text = str(token).replace("\\", "/").strip()
    name = text.rsplit("/", 1)[-1].lower()
    for suffix in (".exe", ".com", ".bat", ".cmd"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


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


# 根目标：rm -rf、Remove-Item -Recurse 删到这些地方就是删整盘。
_ROOT_TARGETS = {"/", "/*", "*", "c:\\", "c:/", "c:"}


def _rm_rf_root(lower_tokens):
    """rm + -r/-f 组合 + 根目标 → 真删整盘。
    拆开的 -r -f 也算（把所有 - 开头的词的字母拼起来看）。"""
    flags = "".join(t[1:] for t in lower_tokens[1:] if t.startswith("-") and len(t) > 1)
    if "r" not in flags or "f" not in flags:
        return False
    return any(t in _ROOT_TARGETS for t in lower_tokens[1:])


def _remove_item_recurse_root(lower_tokens):
    """Remove-Item -Recurse + 根目标 → 真删整盘。"""
    if "-recurse" not in lower_tokens:
        return False
    return any(t in _ROOT_TARGETS for t in lower_tokens[1:])


def _check_tokens(tokens):
    """按词表检查一段命令，返回 (tier, reason)，tier 是 2 / 1 / 0。

    命令头先归一化（去路径、去 .exe/.com/.bat/.cmd 后缀、小写）再比对，
    Tier 2 和 Tier 1 的所有比对都走归一化后的名字：shutdown.exe、
    完整路径、大小写混写都落到同一个名字上，绕不过硬禁。
    """
    if not tokens:
        return 0, ""
    prog = _program_name(tokens[0])
    lower_tokens = [t.lower() for t in tokens]
    text = " ".join(tokens).strip()
    if prog in _TIER2_EXECUTABLES:
        return 2, f"硬禁命令：{tokens[0]}"
    # 危险组合也按归一化后的程序名查：全路径/带后缀的 rd、del、rm、
    # Remove-Item 同样拦住。下面的 _TIER2_SEGMENT_RES 是第二道网
    # （直接比整段文本），两道都留着。
    if prog in {"rd", "rmdir", "del"} and "/s" in lower_tokens:
        return 2, f"硬禁命令：{text}"
    if prog == "rm" and _rm_rf_root(lower_tokens):
        return 2, f"硬禁命令：{text}"
    if prog == "remove-item" and _remove_item_recurse_root(lower_tokens):
        return 2, f"硬禁命令：{text}"
    for pattern in _TIER2_SEGMENT_RES:
        if pattern.search(text):
            return 2, f"硬禁命令：{text}"
    # 解释器内联代码（python -c / -m、node -e 等）：现拼的代码直接跑，
    # 没经过“写文件要确认”那一关，必须当 Tier 1。
    # 注意：内联代码的内容不做静态分析（Python 代码没法用命令规则查），
    # 所以只升到 Tier 1 而不是 Tier 2——确认时 BOOS 能看到完整命令原文，
    # 由他判断；想彻底堵住要靠 OS 层隔离（Box 上测 gVisor）。
    inline_flags = _INTERPRETER_INLINE_FLAGS.get(prog)
    if inline_flags is not None and any(tok in inline_flags for tok in tokens[1:]):
        return 1, "跑解释器内联代码，需要确认"
    # shell 解释器带自由代码：先按 Tier 1 处理，里面的内容另行逐段查。
    if prog in _SHELLS:
        return 1, "跑 shell 自由代码，需要确认"
    if (prog == "pip" or "pip" in lower_tokens) and (
        "install" in lower_tokens or "uninstall" in lower_tokens
    ):
        return 1, "装/卸 Python 包，需要确认"
    if prog == "git":
        if "push" in lower_tokens and ("--force" in lower_tokens or "-f" in lower_tokens):
            return 1, "git 强制推送，需要确认"
        if "reset" in lower_tokens and "--hard" in lower_tokens:
            return 1, "git 硬重置，需要确认"
        if "clean" in lower_tokens and "-f" in lower_tokens:
            return 1, "git 强制清理，需要确认"
    if prog in {"rm", "del", "rmdir", "rd", "remove-item", "unlink"}:
        return 1, "删除文件/目录，需要确认"
    if prog in {"curl", "wget", "invoke-webrequest"}:
        return 1, "联网下载，需要确认"
    if prog in {"npm", "pnpm", "yarn"} and (
        "install" in lower_tokens or "add" in lower_tokens or "i" in lower_tokens
    ):
        return 1, "装 npm 包，需要确认"
    if prog == "reg" and len(lower_tokens) > 1 and lower_tokens[1] in {"add", "delete"}:
        return 1, "改注册表，需要确认"
    return 0, ""


def _check_segment(segment):
    """检查一段命令文本，返回 (tier, reason)：先按空白切词，再按词表查。"""
    return _check_tokens(_tokens(segment))


def _decode_powershell_encoded(text):
    """解 powershell -EncodedCommand 的 base64：PowerShell 默认用
    UTF-16LE 编码命令，解不开就按 UTF-8 试；都解不开返回 None
    （调用方明确标注“解不出来”，不静默放行）。"""
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError:
        return None
    for encoding in ("utf-16-le", "utf-8"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def classify_command(argv):
    """给一条结构化命令分级：返回 (tier, reason)。

    argv 必须是命令列表（["python", "script.py"]），不接受拼好的字符串——
    模型拼 shell 字符串是 GPT 指出的隐患，结构化调用从源头上堵住它。
    分级顺序：先查整条命令的“下载即执行”，再按 argv 原样查命令头
    （带空格的完整路径不能先拼串再切词），再按段逐段查，最后查
    shell -c 里面的自由代码；先硬禁后需确认。
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
    # 先按 argv 原样查一遍命令头：带空格的完整路径（比如
    # C:\Program Files\Git\bin\git.exe）拼成字符串再切词，会从空格处
    # 把路径切断，命令头就认不出来了。
    worst_tier, worst_reason = _check_tokens(normalized)
    if worst_tier == 2:
        return 2, worst_reason
    # 再按 | && || ; 拆段逐段查（shell 链式命令、内嵌的第二条命令）。
    segments = _split_segments(joined)
    for segment in segments:
        tier, reason = _check_segment(segment)
        if tier > worst_tier:
            worst_tier, worst_reason = tier, reason
            if tier == 2:
                return 2, reason
    # sh/bash/cmd/powershell -c "..."：把引号里的内容也当命令查。
    # 命令头同样先归一化（全路径的 cmd.exe 也认），-Command 大小写不敏感。
    if (
        _program_name(normalized[0]) in _SHELLS
        and len(normalized) > 2
        and normalized[1].lower() in ("-c", "-command", "/c")
    ):
        inner = normalized[2]
        for segment in _split_segments(inner):
            tier, reason = _check_segment(segment)
            if tier == 2:
                return 2, reason
            if tier > worst_tier:
                worst_tier, worst_reason = tier, reason
    # powershell -EncodedCommand/-enc：base64 包起来的命令也要拆开查。
    # 解出来的内容按同一套规则查（比如编进去一个 shutdown 也进 Tier 2）；
    # 解出来的原命令（截断到 200 字）写进确认理由里，让 BOOS 点头时
    # 看得懂里面是什么，而不是对着一串 base64 点头。解不开就明确标注
    #“解不出来”，内容未检查。
    if (
        _program_name(normalized[0]) in _SHELLS
        and len(normalized) > 2
        and normalized[1].lower() in ("-encodedcommand", "-enc")
    ):
        decoded = _decode_powershell_encoded(normalized[2])
        if decoded is None:
            # shell 本身已经是 Tier 1，这里只换理由：让 BOOS 知道这串
            # base64 解不出来、内容没检查过，不能让他对着看不懂的东西点头。
            worst_tier, worst_reason = (
                1,
                "内含编码命令（-EncodedCommand），解不出来，内容未检查，需要确认",
            )
        else:
            for segment in _split_segments(decoded):
                tier, reason = _check_segment(segment)
                if tier == 2:
                    return 2, reason
                if tier > worst_tier:
                    worst_tier, worst_reason = tier, reason
            shown = decoded if len(decoded) <= 200 else decoded[:200] + "……"
            if worst_tier <= 1:
                worst_tier, worst_reason = 1, f"跑 shell 编码命令，需要确认：{shown}"
    if worst_tier == 1:
        return 1, worst_reason
    return 0, ""


# ---------- 确认有效性（十一的做法） ----------

# 确认必须当场、当次、来自 BOOS 本人：高风险操作要 BOOS 明确点头才行。
# 工具层自己没法替 BOOS 点头，所以 confirmed 的含义是调用方的担保，
# 不是工具层独立核实过的事实：
#   - True（布尔值）：调用方担保“BOOS 在本轮、当场、针对这一次操作
#     点过头了”。这是兼容旧调用的写法，审计里记为 caller-attested；
#     调用方（聊天层/命令行）必须每次现问现拿，绝不能拿上一次的确认
#     顶这一次的，也不能默认传 True。
#   - dict {"by": ..., "scope": ...}：带作用域的确认，scope 必须和
#     expected_scope（本次操作的精确描述，比如 "run:python -c ..."、
#     "write:notes/a.txt"）完全一致才算数；对不上就拒绝——
#     一次确认只能顶一次操作，这是推荐的写法。
# 返回 (status, reason)：status 是 "user-scoped" / "caller-attested" /
# None（None 表示没拿到有效确认）；reason 是拒绝时的原因。


def resolve_confirmation(confirmed, expected_scope):
    """判断一次确认是不是有效：返回 (status, reason)。"""
    if isinstance(confirmed, dict):
        scope = confirmed.get("scope")
        if scope is None:
            return None, "确认里没有写清是针对哪个操作，拒绝"
        if str(scope) != str(expected_scope):
            return None, "确认和本次操作对不上，拒绝（一次确认只顶一次操作）"
        return "user-scoped", ""
    if confirmed is True:
        return "caller-attested", ""
    return None, "需要 BOOS 确认后才能做"


def confirmation_scope_for_command(argv):
    """给一条命令算确认作用域：调用方拿 BOOS 的点头时，就按这个格式写 scope。"""
    return "run:" + " ".join(str(arg) for arg in argv)


def confirmation_scope_for_file(action, rel_path):
    """给一个文件操作算确认作用域：action 是 write / edit / delete。"""
    return f"{action}:{rel_path}"
