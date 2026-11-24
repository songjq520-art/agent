"""八项体检、报告变化比较和每周一次补跑。"""
from __future__ import annotations
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid
from .core import Finding, Unconnected, emit, redact, utcnow, confined
from .gates import Gates, GateConfigError, copy_repo

ITEMS = ("记忆", "代码", "服务状态", "模型接口", "身体资源", "备份", "安全", "调度")


_COPY_DIR = re.compile(r"^[a-z]+-[0-9a-f]{32}$")


class Healthcheck:
    def __init__(self, config, adapter=None):
        config.validate()
        config.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        config.archive.chmod(0o700)
        self.config = config
        self.adapter = adapter or Unconnected()

    def check(self):
        c, a = self.config, self.adapter
        def memory():
            data = a.memory(sample_size=20)
            required = ("contradictions", "sources", "duplicates", "buried", "core_integrity")
            if data.get("sample_count") != 20 or any(k not in data for k in required):
                return {**data, "ok": False, "reason": "记忆契约不完整或未抽查20条"}
            return data

        def code():
            gate = Gates(c)
            gate.preflight()
            return gate.validate(copy_repo(c, "check"), [])

        def resources():
            free = shutil.disk_usage(c.repo).free
            extra = a.resources()
            return {**extra, "disk_free_bytes": free,
                    "ok": extra.get("ok") is True and free >= c.disk_min_bytes}

        def security():
            # 仅检查名称是否存在，绝不读取环境变量值。
            present = {name: name in os.environ for name in c.secret_env_names}
            permissions = []
            for name in c.permission_paths:
                p = confined(c.repo, name)
                if not p.exists():
                    permissions.append({"path": name, "error": "不存在"})
                elif os.name != "nt" and stat.S_IMODE(p.stat().st_mode) & 0o077:
                    # 移植注记（2026-11-24）：POSIX 权限位在 Windows 上
                    # 无对应语义（普通文件恒报 0o666），只在 POSIX 判异常；
                    # Windows 侧只确认存在，ACL 检查不在本模块范围。
                    permissions.append({"path": name, "error": "组或其他用户可访问"})
            return {"ok": all(present.values()) and not permissions,
                    "presence": [{"name": name, "present": value} for name, value in present.items()],
                    "permission_anomalies": permissions}

        functions = (memory, code, a.services, a.models, resources, a.backups, security, a.schedule)
        findings = []
        for item, function in zip(ITEMS, functions):
            try:
                details = function()
                required = {
                    "服务状态": ("processes", "endpoints", "recent_logs"),
                    "模型接口": ("judge", "chat", "month_spent", "month_limit"),
                    "身体资源": ("cpu_percent", "memory_percent", "database_bytes", "ledger_bytes"),
                    "备份": ("last_backup", "integrity"),
                    "调度": ("maintenance", "pending_questions"),
                }.get(item, ())
                if any(key not in details for key in required):
                    details = {**details, "ok": False, "contract_error": "缺少必填检查证据"}
                if item == "模型接口" and all(k in details for k in ("judge", "chat", "month_spent", "month_limit")):
                    details["ok"] = (details.get("ok") is True and details["judge"] is True and details["chat"] is True
                                     and 0 <= details["month_spent"] < details["month_limit"])
                if item == "备份" and details.get("integrity") is not True:
                    details["ok"] = False
                ok = details.get("ok") is True
                if item == "代码":
                    ok = bool(details.get("quick", {}).get("ok") and (details.get("slow") or {}).get("ok"))
                findings.append(Finding(item, "正常" if ok else "异常",
                                        "info" if ok else details.get("severity", "warning"), details))
            except GateConfigError as exc:
                findings.append(Finding(item, "异常", "critical", {"kind": "gate_configuration_error", "error": str(exc)}))
            except Exception as exc:
                findings.append(Finding(item, "异常", "warning", {"kind": "adapter_error", "error": str(exc)}))
        return redact([dataclasses.asdict(f) for f in findings])

    def _sandbox_copies(self):
        """存档里现有的整仓副本目录：仅 sandboxes/ 直属、名字符合
        「标签-32 位十六进制」规律的真目录；符号链接与其他条目一律不算。"""
        root = self.config.archive / "sandboxes"
        if not root.is_dir():
            return []
        copies = []
        for entry in root.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            if not _COPY_DIR.match(entry.name):
                continue
            copies.append(entry)
        # Windows 的 st_ctime 是创建时间，即运行先后；名字仅作同刻兜底排序。
        copies.sort(key=lambda p: (p.stat().st_ctime, p.name))
        return copies

    @staticmethod
    def _delete_copy(path):
        try:
            shutil.rmtree(path)
        except OSError:
            # 副本里可能有只读文件（.venv 等）：放宽权限后再试一次。
            for dirpath, dirnames, filenames in os.walk(path):
                for name in dirnames + filenames:
                    try:
                        os.chmod(Path(dirpath) / name, 0o700)
                    except OSError:
                        pass
            try:
                os.chmod(path, 0o700)
            except OSError:
                pass
            shutil.rmtree(path)

    def prune_sandbox_copies(self, protect=()):
        """副本保留策略（BOOS 2026-11-24 定：只留最近若干份，默认 3）。

        只处理 _sandbox_copies() 认定的整仓副本；体检报告、weekly 记录
        和存档里其他任何条目都不碰。删除前复核路径严格落在 sandboxes
        根内；protect 里的（本次运行刚建的副本）绝不删。清理失败只记
        进返回结果、写进体检报告备注，绝不抛错拖垮体检本身，下轮再试。
        """
        result = {"keep": self.config.archive_keep_copies, "removed": [],
                  "errors": [], "remaining": None}
        try:
            root = self.config.archive / "sandboxes"
            copies = self._sandbox_copies()
            protected = {Path(p).name for p in protect}
            excess = len(copies) - result["keep"]
            for path in copies:
                if excess <= 0:
                    break
                if path.name in protected:
                    continue
                if path.resolve().parent != root.resolve():
                    result["errors"].append({"name": path.name, "error": "路径不在存档副本根内，已跳过"})
                    continue
                try:
                    self._delete_copy(path)
                    result["removed"].append(path.name)
                    excess -= 1
                except OSError as exc:
                    result["errors"].append({"name": path.name, "error": str(exc)})
            result["remaining"] = len(self._sandbox_copies())
        except Exception as exc:
            result["errors"].append({"name": None, "error": str(exc)})
        return result
    def report(self, repair=None):
        known = {p.name for p in self._sandbox_copies()}
        findings = self.check()
        created = [p for p in self._sandbox_copies() if p.name not in known]
        retention = self.prune_sandbox_copies(protect=created)
        prior = None
        report_dir = self.config.archive / "reports"
        for path in sorted(report_dir.glob("*.json"), reverse=True):
            try:
                candidate = json.loads(path.read_text(encoding="utf-8"))
                if candidate.get("kind") == "healthcheck":
                    prior = candidate
                    break
            except (ValueError, OSError):
                continue
        previous = {f["item"]: f for f in prior["findings"]} if prior else {}
        changes = [{"item": f["item"], "before": previous.get(f["item"]), "after": f}
                   for f in findings if previous.get(f["item"]) != f]
        stamp = utcnow().strftime("%Y%m%dT%H%M%S.%fZ")
        data = {"kind": "healthcheck", "time": utcnow().isoformat(),
                "headlines": [f for f in findings if f["severity"] == "critical"],
                "findings": findings, "changes": changes, "previous_time": prior.get("time") if prior else None,
                "repair": repair, "sandbox_retention": retention,
                "connection_note": "适配器未接通时标异常，不冒充实连"}
        path = report_dir / (stamp + "-" + uuid.uuid4().hex + ".json")
        emit(path, data)
        return path, redact(data)

    def weekly(self, now=None):
        now = now or utcnow()
        previous = []
        for path in (self.config.archive / "weekly").glob("*.json"):
            try:
                previous.append(dt.datetime.fromisoformat(json.loads(path.read_text())["completed_at"]))
            except (KeyError, ValueError, OSError):
                pass
        if previous and (now - max(previous)).total_seconds() < self.config.weekly_interval_seconds:
            return {"ran": False, "reason": "未逾期"}
        # 十二调度器提供持久化互斥租约，避免多进程重复补跑。
        if not self.adapter.weekly_lease("healthcheck-weekly"):
            return {"ran": False, "reason": "未取得十二调度租约或尚未接通"}
        path, data = self.report()
        emit(self.config.archive / "weekly" / (uuid.uuid4().hex + ".json"),
             {"completed_at": now.isoformat(), "report": str(path)})
        return {"ran": True, "report": str(path)}
