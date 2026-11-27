# tests/test_skill_format.py（Agent Skills 公开规范：解析/校验/发现/激活）
# 全部离线：技能包都是在 tmp_path 里现造的假数据，不联网、不产生费用。

import dataclasses

import pytest

from skills.registry import SkillIndexEntry, SkillRegistry
from skills.skill_format import (
    SkillFormatError,
    load_skill_md,
    parse_skill_md,
    read_frontmatter,
)

VALID_SKILL_MD = """---
name: pdf-tools
description: 处理 PDF 文件：合并、拆分、提取文字。用户提到 PDF 时使用。
license: MIT
compatibility: 需要 Python 3.10 以上
metadata:
  author: test
  version: "1.0"
allowed-tools: read_file run_python
---

# PDF 工具

第一步：先看清楚用户要合并还是拆分。
"""


def make_skill(root, dir_name, content=VALID_SKILL_MD, extra_files=None):
    """在 root 下造一个技能目录，返回目录路径。"""
    skill_dir = root / dir_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for rel, text in (extra_files or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return skill_dir


def skill_md(frontmatter: str, body: str = "正文") -> str:
    return f"---\n{frontmatter}\n---\n\n{body}\n"


# ---------- 正例：完整合法技能包 ----------


def test_valid_full_skill_parses_all_fields():
    metadata, body = parse_skill_md(VALID_SKILL_MD, dir_name="pdf-tools")
    assert metadata.name == "pdf-tools"
    assert metadata.description.startswith("处理 PDF 文件")
    assert metadata.license == "MIT"
    assert metadata.compatibility == "需要 Python 3.10 以上"
    assert metadata.metadata == {"author": "test", "version": "1.0"}
    assert metadata.allowed_tools == ["read_file", "run_python"]
    assert "第一步" in body


def test_minimal_skill_only_required_fields():
    metadata, _ = parse_skill_md(
        skill_md("name: tiny\ndescription: 最小技能"), dir_name="tiny"
    )
    assert metadata.name == "tiny"
    assert metadata.license is None
    assert metadata.metadata == {}
    assert metadata.allowed_tools == []


def test_crlf_line_endings_parse():
    # 笔记本是 Windows，SKILL.md 可能是 \r\n 换行。
    content = VALID_SKILL_MD.replace("\n", "\r\n")
    metadata, body = parse_skill_md(content, dir_name="pdf-tools")
    assert metadata.name == "pdf-tools"
    assert "第一步" in body


# ---------- 发现与激活：渐进加载 ----------


def test_discover_index_has_no_body(tmp_path):
    make_skill(
        tmp_path,
        "pdf-tools",
        extra_files={
            "scripts/run.py": "print('hi')",
            "references/REF.md": "参考",
            "assets/tpl.txt": "模板",
        },
    )
    registry = SkillRegistry()
    found = registry.discover(tmp_path)
    assert [entry.name for entry in found] == ["pdf-tools"]
    entry = registry.get("pdf-tools")
    assert entry.description.startswith("处理 PDF 文件")
    # 索引条目结构上就没有正文字段——渐进加载第一层不碰正文。
    assert {f.name for f in dataclasses.fields(SkillIndexEntry)} == {
        "name",
        "description",
        "path",
    }
    assert registry.errors == {}


def test_discover_ignores_non_skill_dirs_and_files(tmp_path):
    (tmp_path / "not-a-skill").mkdir()
    (tmp_path / "loose.txt").write_text("散文件", encoding="utf-8")
    make_skill(tmp_path, "pdf-tools")
    registry = SkillRegistry()
    assert [entry.name for entry in registry.discover(tmp_path)] == ["pdf-tools"]


def test_discover_records_invalid_skill_without_blocking_others(tmp_path):
    make_skill(tmp_path, "pdf-tools")
    make_skill(tmp_path, "bad-one", content=skill_md("name: BAD\ndescription: x"))
    registry = SkillRegistry()
    assert [entry.name for entry in registry.discover(tmp_path)] == ["pdf-tools"]
    assert "bad-one" in registry.errors
    assert "name" in registry.errors["bad-one"]


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(SkillFormatError, match="不存在"):
        SkillRegistry().discover(tmp_path / "nope")


def test_activate_returns_body_and_resource_list(tmp_path):
    make_skill(
        tmp_path,
        "pdf-tools",
        extra_files={
            "scripts/run.py": "print('hi')",
            "references/REF.md": "参考内容",
            "assets/tpl.txt": "模板内容",
        },
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    loaded = registry.activate("pdf-tools")
    assert "第一步" in loaded.body
    assert loaded.metadata.name == "pdf-tools"
    # 资源只列相对路径名，不带内容；SKILL.md 本身不在清单里。
    assert loaded.resources == [
        "assets/tpl.txt",
        "references/REF.md",
        "scripts/run.py",
    ]


def test_activate_unknown_skill_raises(tmp_path):
    make_skill(tmp_path, "pdf-tools")
    registry = SkillRegistry()
    registry.discover(tmp_path)
    with pytest.raises(SkillFormatError, match="没有这个技能"):
        registry.activate("no-such-skill")


def test_read_frontmatter_ignores_body(tmp_path):
    # 正文里放会让 YAML 解析炸掉的内容，只读 frontmatter 也必须成功——
    # 证明发现阶段确实不读正文。
    content = skill_md(
        "name: pdf-tools\ndescription: 处理 PDF", body="name: [未闭合\n: : :"
    )
    skill_dir = make_skill(tmp_path, "pdf-tools", content=content)
    metadata = read_frontmatter(skill_dir / "SKILL.md")
    assert metadata.name == "pdf-tools"


# ---------- 校验反例：每条规则一个 ----------


def test_missing_frontmatter_rejected():
    with pytest.raises(SkillFormatError, match="frontmatter"):
        parse_skill_md("# 没有 frontmatter\n正文", dir_name="x")


def test_unclosed_frontmatter_rejected():
    with pytest.raises(SkillFormatError, match="没有收尾"):
        parse_skill_md("---\nname: x\ndescription: y\n", dir_name="x")


def test_invalid_yaml_rejected():
    with pytest.raises(SkillFormatError, match="合法 YAML"):
        parse_skill_md(skill_md("name: [未闭合"), dir_name="x")


def test_frontmatter_not_a_mapping_rejected():
    with pytest.raises(SkillFormatError, match="键值映射"):
        parse_skill_md(skill_md("- 只是一个列表"), dir_name="x")


def test_missing_name_rejected():
    with pytest.raises(SkillFormatError, match="name"):
        parse_skill_md(skill_md("description: 有描述没名字"), dir_name="x")


def test_missing_description_rejected():
    with pytest.raises(SkillFormatError, match="description"):
        parse_skill_md(skill_md("name: some-skill"), dir_name="some-skill")


def test_empty_description_rejected():
    with pytest.raises(SkillFormatError, match="不能为空"):
        parse_skill_md(
            skill_md('name: some-skill\ndescription: "  "'), dir_name="some-skill"
        )


@pytest.mark.parametrize(
    "bad_name",
    [
        "PDF-Tools",  # 大写
        "-pdf-tools",  # 首连字符
        "pdf-tools-",  # 尾连字符
        "pdf--tools",  # 连续连字符
        "pdf_tools",  # 下划线
        "pdf tools",  # 空格
    ],
)
def test_bad_name_shapes_rejected(bad_name):
    with pytest.raises(SkillFormatError, match="name"):
        parse_skill_md(
            skill_md(f"name: {bad_name}\ndescription: 描述"), dir_name=bad_name
        )


def test_name_too_long_rejected():
    long_name = "a" * 65
    with pytest.raises(SkillFormatError, match="64"):
        parse_skill_md(
            skill_md(f"name: {long_name}\ndescription: 描述"), dir_name=long_name
        )


def test_name_must_match_directory():
    with pytest.raises(SkillFormatError, match="目录名"):
        parse_skill_md(
            skill_md("name: pdf-tools\ndescription: 描述"), dir_name="other-dir"
        )


def test_description_too_long_rejected():
    long_desc = "描" * 1025
    with pytest.raises(SkillFormatError, match="1024"):
        parse_skill_md(
            skill_md(f"name: some-skill\ndescription: {long_desc}"),
            dir_name="some-skill",
        )


def test_compatibility_too_long_rejected():
    long_compat = "x" * 501
    with pytest.raises(SkillFormatError, match="500"):
        parse_skill_md(
            skill_md(
                "name: some-skill\ndescription: 描述\n"
                f"compatibility: {long_compat}"
            ),
            dir_name="some-skill",
        )


def test_metadata_value_must_be_string():
    with pytest.raises(SkillFormatError, match="metadata"):
        parse_skill_md(
            skill_md("name: some-skill\ndescription: 描述\nmetadata:\n  count: 3"),
            dir_name="some-skill",
        )


def test_metadata_must_be_mapping():
    with pytest.raises(SkillFormatError, match="metadata"):
        parse_skill_md(
            skill_md("name: some-skill\ndescription: 描述\nmetadata: 不是映射"),
            dir_name="some-skill",
        )


def test_allowed_tools_wrong_type_rejected():
    with pytest.raises(SkillFormatError, match="allowed-tools"):
        parse_skill_md(
            skill_md("name: some-skill\ndescription: 描述\nallowed-tools: 42"),
            dir_name="some-skill",
        )


# ---------- 边界值与按路径加载 ----------


def test_name_exactly_64_chars_ok():
    name64 = "a" * 64
    metadata, _ = parse_skill_md(
        skill_md(f"name: {name64}\ndescription: 描述"), dir_name=name64
    )
    assert metadata.name == name64


def test_description_exactly_1024_chars_ok():
    desc1024 = "描" * 1024
    metadata, _ = parse_skill_md(
        skill_md(f"name: some-skill\ndescription: {desc1024}"),
        dir_name="some-skill",
    )
    assert metadata.description == desc1024


def test_empty_name_rejected():
    with pytest.raises(SkillFormatError, match="name"):
        parse_skill_md(
            skill_md('name: ""\ndescription: 描述'), dir_name="x"
        )


def test_load_skill_md_from_path(tmp_path):
    skill_dir = make_skill(tmp_path, "pdf-tools")
    metadata, body = load_skill_md(skill_dir / "SKILL.md")
    assert metadata.name == "pdf-tools"
    assert "第一步" in body


def test_activate_excludes_pycache_from_resources(tmp_path):
    make_skill(
        tmp_path,
        "pdf-tools",
        extra_files={
            "scripts/run.py": "print('hi')",
            "scripts/__pycache__/run.cpython-312.pyc": "缓存假数据",
        },
    )
    registry = SkillRegistry()
    registry.discover(tmp_path)
    loaded = registry.activate("pdf-tools")
    assert loaded.resources == ["scripts/run.py"]
