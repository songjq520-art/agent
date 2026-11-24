"""提案子进程入口（路线①）：只被独立解释器当脚本执行，父进程从不 import。

用法：python proposer_child.py <task.json> <result.json>

task.json：{"spec": "模块路径:函数名", "evidence": {...},
            "cwd": "...", "sys_path": [...]}
行为：插入 sys_path → chdir(cwd) → import 具名函数 → 以 evidence
为唯一参数调用 → 结果写 result.json（先写 .tmp 再 os.replace 原子
落盘）后以 0 退出。函数抛异常 → 写 {"status": "error", ...} 仍以
0 退出（父进程据此与「崩溃/非零退出」区分）；函数若 os._exit(非零)
或硬崩溃，则没有结果文件、退出码非零，父进程走崩溃路径。

具名函数契约：接收 evidence dict，返回 None（无提案）或候选——
Candidate 鸭子类型（有 files/explanation 属性）或等价 dict
{"files": {...}, "explanation": str, "cost": float, "calls": int}。
"""
import importlib
import json
import os
import sys
from pathlib import Path


def main(argv):
    task = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    result_path = argv[2]
    for entry in reversed(task.get("sys_path") or []):
        sys.path.insert(0, entry)
    if task.get("cwd"):
        os.chdir(task["cwd"])
    module_name, _, func_name = task["spec"].partition(":")
    try:
        fn = getattr(importlib.import_module(module_name), func_name)
        value = fn(task["evidence"])
        if value is None:
            outcome = {"status": "ok", "candidate": None}
        elif hasattr(value, "files") and hasattr(value, "explanation"):
            candidate = {
                "files": dict(value.files),
                "explanation": str(value.explanation),
                "cost": float(getattr(value, "cost", 0.0)),
                "calls": int(getattr(value, "calls", 1)),
            }
            json.dumps(candidate)
            outcome = {"status": "ok", "candidate": candidate}
        elif isinstance(value, dict):
            candidate = {
                "files": value["files"],
                "explanation": value.get("explanation", ""),
                "cost": value.get("cost", 0.0),
                "calls": value.get("calls", 1),
            }
            json.dumps(candidate)
            outcome = {"status": "ok", "candidate": candidate}
        else:
            raise TypeError(f"提案返回值类型不支持：{type(value).__name__}")
    except Exception as exc:
        outcome = {"status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
    tmp = result_path + ".tmp"
    Path(tmp).write_text(json.dumps(outcome, ensure_ascii=False),
                         encoding="utf-8")
    os.replace(tmp, result_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
