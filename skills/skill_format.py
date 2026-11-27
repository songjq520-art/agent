# skills/skill_format.py（技能格式：SKILL.md 解析与校验）
# 作用：按 Agent Skills 公开规范（https://agentskills.io/specification，
#       2026-11-26 读规范原文核对）解析技能包里的 SKILL.md，
#       并逐条校验 YAML frontmatter，违规给出明确错误。
# 规范要点：
#   - SKILL.md = YAML frontmatter（--- 包起来）+ Markdown 正文；
#   - name 必填：1-64 字符，只许小写字母、数字、连字符，
#     不许首尾是连字符、不许连续连字符，且必须与技能目录名一致；
#   - description 必填：1-1024 字符，写清「做什么、何时用」；
#   - 可选：license（许可证名）、compatibility（环境要求，≤500 字符）、
#     metadata（字符串→字符串的键值映射）、
#     allowed-tools（空格分隔的预批准工具名单，规范标注为实验字段）；
#   - 渐进加载三层：① 启动时只读 name+description 建索引；
#     ② 技能激活时才读全文正文；③ scripts/、references/、assets/
#     里的资源到执行时才取。本模块管解析与校验这一层，
#     发现/激活在 skills/registry.py。
# 关系：只依赖 PyYAML（已在 requirements.txt 里声明）；不依赖仓库其他模块。

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

NAME_MAX = 64
DESCRIPTION_MAX = 1024
COMPATIBILITY_MAX = 500

# name 的形状：小写字母/数字打头的段，段之间单个连字符连接。
# 这个正则同时挡住：大写、空格、下划线等非法字符，首尾连字符，连续连字符。
_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class SkillFormatError(ValueError):
    """SKILL.md 不合规范。消息里写清哪条规则、实际是什么。"""


@dataclass
class SkillMetadata:
    """SKILL.md frontmatter 解析后的元数据（渐进加载第一层就用这些）。"""

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: list[str] = field(default_factory=list)


def split_frontmatter(text: str) -> tuple[str, str]:
    """把 SKILL.md 全文拆成 (frontmatter 的 YAML 文本, Markdown 正文)。

    文件必须以 --- 开头、再以一行 --- 收尾；中间就是 frontmatter。
    兼容 Windows 的 \\r\\n 换行（笔记本是 Windows，文件可能带 \\r）。
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillFormatError(
            "SKILL.md 缺少 YAML frontmatter：文件第一行必须是 ---"
        )
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :])
    raise SkillFormatError(
        "SKILL.md 的 frontmatter 没有收尾：找不到第二行 ---"
    )


def _validate_metadata(raw: dict, dir_name: str | None) -> SkillMetadata:
    """把 frontmatter 的原始映射逐条按规范校验，转成 SkillMetadata。"""
    name = raw.get("name")
    if name is None:
        raise SkillFormatError("frontmatter 缺少必填字段 name")
    if not isinstance(name, str):
        raise SkillFormatError(f"name 必须是字符串，实际是 {type(name).__name__}")
    if len(name) > NAME_MAX:
        raise SkillFormatError(
            f"name 超过 {NAME_MAX} 字符上限（实际 {len(name)} 字符）：{name!r}"
        )
    if not _NAME_RE.match(name):
        raise SkillFormatError(
            "name 只能含小写字母、数字和连字符，且不许首尾是连字符、"
            f"不许连续连字符，实际是：{name!r}"
        )
    if dir_name is not None and name != dir_name:
        raise SkillFormatError(
            f"name 必须与技能目录名一致：frontmatter 里是 {name!r}，"
            f"目录名是 {dir_name!r}"
        )

    description = raw.get("description")
    if description is None:
        raise SkillFormatError("frontmatter 缺少必填字段 description")
    if not isinstance(description, str):
        raise SkillFormatError(
            f"description 必须是字符串，实际是 {type(description).__name__}"
        )
    if not description.strip():
        raise SkillFormatError("description 不能为空")
    if len(description) > DESCRIPTION_MAX:
        raise SkillFormatError(
            f"description 超过 {DESCRIPTION_MAX} 字符上限"
            f"（实际 {len(description)} 字符）"
        )

    license_value = raw.get("license")
    if license_value is not None and not isinstance(license_value, str):
        raise SkillFormatError(
            f"license 必须是字符串，实际是 {type(license_value).__name__}"
        )

    compatibility = raw.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            raise SkillFormatError(
                "compatibility 必须是字符串，实际是 "
                f"{type(compatibility).__name__}"
            )
        if len(compatibility) > COMPATIBILITY_MAX:
            raise SkillFormatError(
                f"compatibility 超过 {COMPATIBILITY_MAX} 字符上限"
                f"（实际 {len(compatibility)} 字符）"
            )

    metadata_raw = raw.get("metadata")
    metadata: dict[str, str] = {}
    if metadata_raw is not None:
        if not isinstance(metadata_raw, dict):
            raise SkillFormatError(
                "metadata 必须是键值映射，实际是 "
                f"{type(metadata_raw).__name__}"
            )
        for key, value in metadata_raw.items():
            if not isinstance(value, str):
                raise SkillFormatError(
                    f"metadata 的值必须是字符串：键 {key!r} 的值是 "
                    f"{type(value).__name__}"
                )
            metadata[str(key)] = value

    allowed_raw = raw.get("allowed-tools")
    allowed_tools: list[str] = []
    if allowed_raw is not None:
        if isinstance(allowed_raw, str):
            allowed_tools = allowed_raw.split()
        elif isinstance(allowed_raw, list) and all(
            isinstance(item, str) for item in allowed_raw
        ):
            # 规范写的是空格分隔字符串；顺手兼容 YAML 列表写法。
            allowed_tools = list(allowed_raw)
        else:
            raise SkillFormatError(
                "allowed-tools 必须是空格分隔的字符串，实际是 "
                f"{type(allowed_raw).__name__}"
            )

    return SkillMetadata(
        name=name,
        description=description,
        license=license_value,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=allowed_tools,
    )


def parse_frontmatter(frontmatter_text: str, dir_name: str | None = None) -> SkillMetadata:
    """解析并校验 frontmatter 的 YAML 文本。dir_name 传了就校验 name 与目录名一致。"""
    try:
        raw = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise SkillFormatError(f"frontmatter 不是合法 YAML：{exc}") from exc
    if raw is None:
        raise SkillFormatError("frontmatter 是空的：至少要填 name 和 description")
    if not isinstance(raw, dict):
        raise SkillFormatError(
            f"frontmatter 必须是键值映射，实际是 {type(raw).__name__}"
        )
    return _validate_metadata(raw, dir_name)


def parse_skill_md(text: str, dir_name: str | None = None) -> tuple[SkillMetadata, str]:
    """解析 SKILL.md 全文，返回 (元数据, 正文)。"""
    frontmatter_text, body = split_frontmatter(text)
    return parse_frontmatter(frontmatter_text, dir_name), body


def read_frontmatter(path: Path) -> SkillMetadata:
    """只读 SKILL.md 的 frontmatter 建索引用：读到收尾的 --- 就停，正文不读。

    渐进加载第一层靠这个：技能再多、每个正文再长，启动索引也只碰开头几行。
    """
    path = Path(path)
    collected: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        first = handle.readline()
        if first.strip() != "---":
            raise SkillFormatError(
                f"{path} 缺少 YAML frontmatter：文件第一行必须是 ---"
            )
        for line in handle:
            if line.strip() == "---":
                break
            collected.append(line.rstrip("\r\n"))
        else:
            raise SkillFormatError(
                f"{path} 的 frontmatter 没有收尾：找不到第二行 ---"
            )
    return parse_frontmatter("\n".join(collected), dir_name=path.parent.name)


def load_skill_md(path: Path) -> tuple[SkillMetadata, str]:
    """读 SKILL.md 全文并解析（激活时用），返回 (元数据, 正文)。"""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    return parse_skill_md(text, dir_name=path.parent.name)
