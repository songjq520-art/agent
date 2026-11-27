# skills/__init__.py（技能包）
# 作用：让 skills/ 成为一个 Python 包，和 memory/、tools/ 一样，
#       这样可以用 from skills.registry import SkillRegistry 来用它。
# 内容：按 Agent Skills 公开规范（https://agentskills.io/specification）
#       实现技能包（SKILL.md）的解析校验、发现索引与激活加载。
#       目前是自成模块，不接聊天/主流程；接线等以后另行立项。
