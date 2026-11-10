# tests/test_tools_cli.py（第一批工具：命令行入口测试）
# 作用：验证 tools/cli.py 的 cleanup 流程走真实的 CLI 函数和真实文件——
# 扫描必须带钥匙生成 HMAC 预览（环境变量优先，其次钥匙文件，没有就自动生成），
# 删除必须用同一把钥匙验；篡改清单后重算普通 SHA-256 也必须被拒绝，原文件保留。
# 全部离线，只碰临时目录，不产生费用。

import hashlib
import json
from pathlib import Path

import pytest

from tools.cli import main
from tools.errors import ToolError


class _FakeOutcome:
    """cmd_search 测试用的假搜索结果：不联网。"""

    status = "ok"

    def to_dict(self):
        return {"status": "ok", "results": []}


class _FakeSearchService:
    """cmd_search 测试用的假调度器：记下传进来的 translator，不联网。"""

    last = None

    def __init__(self, translator=None):
        self.translator = translator
        _FakeSearchService.last = self

    def search(self, *args, **kwargs):
        return _FakeOutcome()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    # 真实临时目录：扫这里，删这里；工作区也放临时目录，审计/回收站不碰别处。
    root = tmp_path / "tree"
    (root / "temp").mkdir(parents=True)
    (root / "temp" / "a.tmp").write_bytes(b"x" * 100)
    monkeypatch.setenv("TWELVE_WORKSPACE", str(tmp_path / "ws"))
    monkeypatch.setenv("TWELVE_CLEANUP_KEY", "test-cleanup-key")
    # 钥匙文件指到临时目录：测试绝不碰项目根目录下真的 .cleanup/cleanup_key。
    monkeypatch.setenv("TWELVE_CLEANUP_KEY_FILE", str(tmp_path / "cleanup_key"))
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


def test_cli_scan_generates_key_file_when_no_env_key(tmp_path, monkeypatch):
    # 11 号：没设 TWELVE_CLEANUP_KEY 时，第一次扫描自动生成一把钥匙
    # 存进钥匙文件，以后扫描和删除一直用它；预览是带钥匙的 HMAC。
    root = tmp_path / "tree"
    (root / "temp").mkdir(parents=True)
    (root / "temp" / "a.tmp").write_bytes(b"x" * 100)
    key_file = tmp_path / "cleanup_key"
    monkeypatch.delenv("TWELVE_CLEANUP_KEY", raising=False)
    monkeypatch.setenv("TWELVE_CLEANUP_KEY_FILE", str(key_file))
    monkeypatch.setenv("TWELVE_WORKSPACE", str(tmp_path / "ws"))
    main(["cleanup", "scan", str(root), "--kind", "temp",
          "--out", str(tmp_path / "preview.json")])
    assert key_file.is_file()
    assert key_file.read_text(encoding="utf-8").strip()
    preview = json.loads((tmp_path / "preview.json").read_text(encoding="utf-8"))
    assert preview["checksum_algo"] == "hmac-sha256"
    # 同一把钥匙删得掉：删除时也没设环境变量，走钥匙文件验。
    main(["cleanup", "delete", "--preview", str(tmp_path / "preview.json"),
          "--categories", "temp", "--root", str(root),
          "--trash", str(tmp_path / "trash"), "--yes"])
    assert not (root / "temp" / "a.tmp").exists()


def test_cli_scan_with_env_key_does_not_create_key_file(cli_env):
    # 11 号：设了环境变量钥匙，就用环境变量那把，不生成钥匙文件。
    preview_path = _scan(cli_env)
    assert not (cli_env["tmp"] / "cleanup_key").exists()
    preview = json.loads(preview_path.read_text(encoding="utf-8"))
    assert preview["checksum_algo"] == "hmac-sha256"


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


def test_cli_downgraded_algo_preview_refused_and_files_kept(cli_env):
    # Bug B 回归（CLI 端到端变种）：带钥匙的预览被降级成普通 SHA-256
    #（改 checksum_algo 再重算校验和）——必须拒绝且原文件保留。
    # 注意：这个变种在 CLI 层就被拦住（checksum_algo 不是 hmac-sha256
    # 直接拒绝）；库函数层那道防线由 test_keyed_preview_downgrade_refused 测。
    preview_path = _scan(cli_env)
    downgraded = json.loads(preview_path.read_text(encoding="utf-8"))
    downgraded["checksum_algo"] = "sha256"
    payload = {k: v for k, v in downgraded.items() if k != "checksum"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    downgraded["checksum"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    preview_path.write_text(json.dumps(downgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    with pytest.raises(SystemExit):
        _delete(cli_env, preview_path)
    assert (cli_env["root"] / "temp" / "a.tmp").exists()  # 拒绝是整单的，什么都没删


def test_cli_delete_requires_key(cli_env, monkeypatch):
    # 删除时没钥匙（环境变量没设，钥匙文件也没有）：拒绝删除，原文件保留。
    # 注意：删除这一步不现生成钥匙——现生成的和扫描时那把对不上，验不过。
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


# ---------- 7 号：CLI 接 DeepSeek 翻译器 ----------


def test_deepseek_translate_calls_api(monkeypatch):
    # CLI 的 DeepSeek 翻译走 chat completions，deepseek-flash。
    from tools.cli import _deepseek_translate

    calls = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Guangzhou weather"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        return FakeResponse()

    monkeypatch.setattr("tools.cli.requests.post", fake_post)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    assert _deepseek_translate("广州天气", "en") == "Guangzhou weather"
    assert calls["url"] == "https://api.deepseek.com/chat/completions"
    assert calls["json"]["model"] == "deepseek-flash"
    assert "广州天气" in calls["json"]["messages"][1]["content"]


def test_deepseek_translate_raises_on_http_error(monkeypatch):
    # 翻译调用失败要抛错，调用方（SearchService）会回落原文，不影响搜索本身。
    from tools.cli import _deepseek_translate

    class FakeResponse:
        status_code = 401

    monkeypatch.setattr("tools.cli.requests.post", lambda *a, **k: FakeResponse())
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    with pytest.raises(ToolError):
        _deepseek_translate("广州天气", "en")


def test_cli_search_uses_deepseek_translator_when_key_set(monkeypatch):
    # 设了 DEEPSEEK_API_KEY，CLI 搜索接上翻译器。
    from tools.cli import _deepseek_translate

    monkeypatch.setattr("tools.cli.SearchService", _FakeSearchService)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake")
    main(["search", "广州天气"])
    assert _FakeSearchService.last.translator is _deepseek_translate


def test_cli_search_hints_when_no_key_and_chinese_query(monkeypatch, capsys):
    # 没设 DEEPSEEK_API_KEY，中文查询打印一句提示（stderr，stdout 只留 JSON）。
    monkeypatch.setattr("tools.cli.SearchService", _FakeSearchService)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    main(["search", "广州天气"])
    out = capsys.readouterr()
    assert "提示" in out.err
    assert "AnySearch 中文结果可能不准" in out.err
    assert json.loads(out.out)["status"] == "ok"


def test_cli_search_no_hint_for_english_query_without_key(monkeypatch, capsys):
    # 英文查询不用翻译，没 Key 也不用提示。
    monkeypatch.setattr("tools.cli.SearchService", _FakeSearchService)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    main(["search", "Guangzhou weather"])
    assert capsys.readouterr().err == ""
