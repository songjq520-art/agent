# tests/test_tools_cli.py（第一批工具：命令行入口测试）
# 作用：验证 tools/cli.py 的 cleanup 流程走真实的 CLI 函数和真实文件——
# 扫描必须带钥匙生成 HMAC 预览，删除必须用同一把钥匙验；篡改清单后
# 重算普通 SHA-256 也必须被拒绝，原文件保留。
# 全部离线，只碰临时目录，不产生费用。

import hashlib
import json
from pathlib import Path

import pytest

from tools.cli import main


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    # 真实临时目录：扫这里，删这里；工作区也放临时目录，审计/回收站不碰别处。
    root = tmp_path / "tree"
    (root / "temp").mkdir(parents=True)
    (root / "temp" / "a.tmp").write_bytes(b"x" * 100)
    monkeypatch.setenv("TWELVE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("TWELVE_CLEANUP_KEY", "test-cleanup-key")
    return {"root": root, "tmp": tmp_path}


def _scan(cli_env, out_name="preview.json"):
    preview_path = cli_env["tmp"] / out_name
    main([
        "cleanup", "scan", str(cli_env["root"]),
        "--kind", "temp", "--out", str(preview_path),
    ])
    return preview_path


def _delete(cli_env, preview_path):
    main([
        "cleanup", "delete", "--preview", str(preview_path),
        "--categories", "temp", "--root", str(cli_env["root"]),
        "--trash", str(cli_env["tmp"] / "trash"), "--yes",
    ])


def test_cli_scan_requires_key(tmp_path, monkeypatch):
    # 没设 TWELVE_CLEANUP_KEY：扫描拒绝生成可删除的预览（与其让 BOOS
    # 白走一趟确认流程再被拒，不如扫描这一步就明确拒绝）。
    root = tmp_path / "tree"
    (root / "temp").mkdir(parents=True)
    (root / "temp" / "a.tmp").write_bytes(b"x" * 100)
    monkeypatch.delenv("TWELVE_CLEANUP_KEY", raising=False)
    with pytest.raises(SystemExit):
        main(["cleanup", "scan", str(root), "--kind", "temp",
              "--out", str(tmp_path / "preview.json")])
    assert not (tmp_path / "preview.json").exists()


def test_cli_scan_produces_keyed_preview(cli_env):
    preview_path = _scan(cli_env)
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert preview["checksum_algo"] == "hmac-sha256"
    assert preview["categories"]["temp"]["files"]


def test_cli_hmac_roundtrip_deletes(cli_env, capsys):
    # 正常 HMAC CLI 往返：设好钥匙，扫描、删除，文件真的删掉。
    preview_path = _scan(cli_env)
    capsys.readouterr()  # 清掉扫描的输出，只看删除的汇报
    _delete(cli_env, preview_path)
    report = json.loads(capsys.readouterr().out)
    assert report["deleted"] == 1
    assert not (cli_env["root"] / "temp" / "a.tmp").exists()


def test_cli_tampered_preview_refused_and_files_kept(cli_env):
    # GPT 三审要求的真实 CLI 流程测试：扫描真实临时目录、存 JSON、
    # 篡改清单、重算普通 SHA-256、尝试删除——必须拒绝且原文件保留。
    # 篡改手法：预览之后新增一个文件（不在冻结清单里），把它塞进清单，
    # 大小/修改时间都对得上；如果没有校验和那一关，删的时候是会真删掉的。
    preview_path = _scan(cli_env)
    keep = cli_env["root"] / "temp" / "keep.tmp"
    keep.write_text("别删我" * 20, encoding="utf-8")
    tampered = json.loads(preview_path.read_text(encoding="utf-8"))
    tampered["categories"]["temp"]["files"].append(
        {"path": str(keep), "size": keep.stat().st_size, "mtime": keep.stat().st_mtime}
    )
    # 攻击者不知道钥匙，只能按旧办法重算一个普通 SHA-256。
    payload = {k: v for k, v in tampered.items() if k != "checksum"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    tampered["checksum"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    preview_path.write_text(json.dumps(tampered, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit):
        _delete(cli_env, preview_path)
    assert keep.exists()  # 篡改塞进去的文件没被删
    assert (cli_env["root"] / "temp" / "a.tmp").exists()  # 拒绝是整单的，什么都没删
    # 拒绝本身记进了审计日志（TWELVE_WORKSPACE 下的 .audit/audit.log）。
    audit_log = cli_env["tmp"] / "ws" / ".audit" / "audit.log"
    events = [json.loads(line) for line in audit_log.read_text(encoding="utf-8").splitlines()]
    assert any(
        e["tool"] == "cleanup" and e["success"] is False and "校验和" in e["reason"]
        for e in events
    )


def test_cli_delete_requires_key(cli_env, monkeypatch):
    # 删除时没钥匙：拒绝删除，原文件保留。
    preview_path = _scan(cli_env)
    monkeypatch.delenv("TWELVE_CLEANUP_KEY", raising=False)
    with pytest.raises(SystemExit):
        _delete(cli_env, preview_path)
    assert (cli_env["root"] / "temp" / "a.tmp").exists()


def test_cli_delete_refuses_plain_sha256_preview(cli_env):
    # 旧的普通 SHA-256 预览（没钥匙生成的）：防不了篡改，删除时明确拒绝。
    from tools.cleanup import build_preview, scan

    preview = build_preview(scan(cli_env["root"], root_kind="temp"))
    assert preview["checksum_algo"] == "sha256"
    preview_path = cli_env["tmp"] / "plain.json"
    preview_path.write_text(json.dumps(preview, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit):
        _delete(cli_env, preview_path)
    assert (cli_env["root"] / "temp" / "a.tmp").exists()
