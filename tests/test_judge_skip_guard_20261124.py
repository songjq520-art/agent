# tests/test_judge_skip_guard_20261124.py（修②防伪，2026-11-24）
#
# 修②给 53 个纠正类测试加了 requires_judge 标记：判官未配置
# （没 Key 等）时跳过并写明原因，配置在场时必须真跑。这里钉住
# 跳过判定本身两头都不许假：
# - 没 Key 时判定必须说跳过、且理由点名缺的是 DEEPSEEK_API_KEY；
# - 有 Key（哪怕是假值）时判定必须说不跳过——判官可用性只查
#   配置在不在，不调模型，所以假 Key 也算「已配置」；真跑时
#   判官答不出来是另一回事（运行时降级），不归跳过判定管。
# 判定函数就是 conftest 里夹具用的 judge_skip_reason，按文件
# 路径直接加载 conftest 来测，不复制一份逻辑出来。

import importlib.util
from pathlib import Path

import pytest


def _load_conftest():
    path = Path(__file__).parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("shier_tests_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skip_reason_names_missing_key_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    reason = _load_conftest().judge_skip_reason()
    assert reason, "没 Key 时跳过判定必须给出理由"
    assert "DEEPSEEK_API_KEY" in reason


def test_skip_reason_none_with_key_present(monkeypatch):
    # 假 Key 也算已配置：判定只查在不在，不验真伪、不调模型。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-dummy-not-real")
    assert _load_conftest().judge_skip_reason() is None


@pytest.mark.requires_judge
def test_marked_test_not_skipped_when_judge_configured():
    # 本测试能跑到这行，就证明有判官配置时标记不会误跳；
    # 无 Key 环境里它会被夹具正常跳过（那正是修②要的行为）。
    assert True
