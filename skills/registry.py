# skills/registry.py（技能注册表：发现 → 索引 → 激活）
# 作用：把 Agent Skills 公开规范的渐进加载三层串起来：
#   1. discover(root)：扫 root 的一级子目录，凡含 SKILL.md 的，
#      只读 frontmatter 建轻量索引（name + description），正文一个字不读；
#      校验不过的技能不进索引，错误记在 self.errors（目录名 → 报错），
#      不连累其他技能。
#   2. activate(name)：按名读完整 SKILL.md（元数据 + 正文），并列出技能
#      目录里的附带资源清单（scripts/、references/、assets/ 等，
#      相对路径，只列名、不读内容）——资源到真正执行时才取。
# 关系：只依赖 skills/skill_format.py；不接聊天/主流程，接线另行立项。

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from skills.skill_format import SkillFormatError, SkillMetadata, load_skill_md, read_frontmatter

SKILL_FILE_NAME = "SKILL.md"


@dataclass
class SkillIndexEntry:
    """索引条目：只有 name + description + 技能目录路径，没有正文。"""

    name: str
    description: str
    path: Path


@dataclass
class LoadedSkill:
    """激活后的技能：完整元数据 + 正文 + 资源清单（资源只列名，不含内容）。"""

    metadata: SkillMetadata
    body: str
    path: Path
    resources: list[str] = field(default_factory=list)


class SkillRegistry:
    """技能注册表：先 discover 建索引，再按名 activate。"""

    def __init__(self) -> None:
        self._entries: dict[str, SkillIndexEntry] = {}
        # 目录名 → 校验报错。discover 时校验不过的技能记在这里，不进索引。
        self.errors: dict[str, str] = {}

    def discover(self, root: Path) -> list[SkillIndexEntry]:
        """扫描 root 的一级子目录建索引，返回这次新发现的索引条目（按名排序）。

        只认含 SKILL.md 的子目录；root 下的普通文件、没有 SKILL.md 的
        目录都跳过。同名技能重复 discover 时以最后一次为准。
        """
        root = Path(root)
        if not root.is_dir():
            raise SkillFormatError(f"技能目录不存在或不是目录：{root}")
        found: list[SkillIndexEntry] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_file = child / SKILL_FILE_NAME
            if not skill_file.is_file():
                continue
            try:
                metadata = read_frontmatter(skill_file)
            except (SkillFormatError, OSError) as exc:
                self.errors[child.name] = str(exc)
                continue
            entry = SkillIndexEntry(
                name=metadata.name, description=metadata.description, path=child
            )
            self._entries[entry.name] = entry
            found.append(entry)
        return sorted(found, key=lambda entry: entry.name)

    def index(self) -> list[SkillIndexEntry]:
        """当前索引的全部条目（按名排序）。条目里只有 name+description。"""
        return [self._entries[name] for name in sorted(self._entries)]

    def get(self, name: str) -> SkillIndexEntry | None:
        """按名查索引条目，没有就返回 None。"""
        return self._entries.get(name)

    def activate(self, name: str) -> LoadedSkill:
        """激活技能：读完整 SKILL.md，取回正文，并列出附带资源清单。

        资源清单 = 技能目录下除 SKILL.md 外的全部文件（相对路径排序），
        只列名不读内容，执行时再按需取。
        """
        entry = self._entries.get(name)
        if entry is None:
            raise SkillFormatError(
                f"没有这个技能：{name!r}（先 discover 建索引，或检查名字）"
            )
        metadata, body = load_skill_md(entry.path / SKILL_FILE_NAME)
        resources = sorted(
            str(item.relative_to(entry.path))
            for item in entry.path.rglob("*")
            if item.is_file()
            and item.name != SKILL_FILE_NAME
            # Python 跑过脚本会留 __pycache__/*.pyc，那是缓存不是技能资源。
            and "__pycache__" not in item.parts
            and item.suffix != ".pyc"
        )
        return LoadedSkill(
            metadata=metadata, body=body, path=entry.path, resources=resources
        )
