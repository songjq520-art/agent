# tests/test_tools_cleanup.py（第一批工具：清理垃圾测试）
# 作用：验证 tools/cleanup.py —— 按种类扫描（大白话分类）、密钥文件永不列入、
# 预览冻结（先扫后删，中间新增的文件不删）、高置信度真删干净并核对腾出的空间、
# 拿不准的先进回收站、删除时二次校验路径不越界。
# 全部离线，只碰临时目录，不产生费用。

from pathlib import Path

from tools.cleanup import build_preview, delete_preview, format_summary, scan


def build_tree(root: Path):
    (root / "temp").mkdir(parents=True)
    (root / "temp" / "a.tmp").write_bytes(b"x" * 100)
    (root / "temp" / "b.txt").write_bytes(b"y" * 200)
    (root / "logs").mkdir()
    (root / "logs" / "app.log").write_bytes(b"l" * 300)
    (root / "downloads").mkdir()
    (root / "downloads" / "setup.msi").write_bytes(b"m" * 400)
    (root / "downloads" / "setup.exe").write_bytes(b"e" * 500)
    (root / "downloads" / "photo.jpg").write_bytes(b"p" * 600)
    (root / "downloads" / "big.iso").write_bytes(b"b" * 1500)
    (root / ".env").write_text("K=1", encoding="utf-8")
    (root / "key.pem").write_text("secret", encoding="utf-8")
    return root


def test_scan_categories(tmp_path):
    root = build_tree(tmp_path / "tree")
    result = scan(root, root_kind="other", large_threshold_mb=0.001)  # 1KB 以上算大文件
    cats = result["categories"]
    assert cats["temp"]["file_count"] == 1  # a.tmp
    assert cats["logs"]["file_count"] == 1  # app.log
    assert cats["installers"]["file_count"] == 2  # setup.msi、setup.exe
    assert cats["large_files"]["file_count"] == 1  # big.iso（1500B > 1KB）
    assert cats["unknown"]["file_count"] == 2  # b.txt、photo.jpg
    assert result["scanned"] == 7
    assert result["skipped_secrets"] == 2  # .env、key.pem


def test_scan_never_lists_secrets(tmp_path):
    root = build_tree(tmp_path / "tree")
    result = scan(root, root_kind="other", large_threshold_mb=10)
    all_paths = [
        f["path"] for cat in result["categories"].values() for f in cat["files"]
    ]
    assert not any(".env" in p or ".pem" in p for p in all_paths)


def test_preview_is_frozen(tmp_path):
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    # 预览之后新增的文件，不在冻结清单里，删的时候不会被删掉。
    (root / "temp" / "new.tmp").write_bytes(b"n" * 50)
    trash = tmp_path / "trash"
    report = delete_preview(preview, ["temp"], trash_dir=trash)
    assert report["deleted"] == 1  # 只有预览里的 a.tmp
    assert (root / "temp" / "new.tmp").exists()
    assert not (root / "temp" / "a.tmp").exists()


def test_delete_high_confidence_permanent_and_verified(tmp_path):
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    trash = tmp_path / "trash"
    report = delete_preview(preview, ["temp", "logs"], trash_dir=trash)
    assert report["deleted"] == 2  # a.tmp + app.log（b.txt 是 unknown，没确认不删）
    assert report["freed_bytes"] == 100 + 300
    assert report["failed"] == []
    assert not (root / "temp" / "a.tmp").exists()
    assert not (root / "logs" / "app.log").exists()
    assert (root / "temp" / "b.txt").exists()  # 没确认的种类不动


def test_delete_low_confidence_goes_to_trash(tmp_path):
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    trash = tmp_path / "trash"
    report = delete_preview(preview, ["unknown"], trash_dir=trash)
    assert report["moved_to_trash"] == 3  # b.txt、photo.jpg、big.iso 先进回收站
    assert not (root / "temp" / "b.txt").exists()
    assert not (root / "downloads" / "photo.jpg").exists()
    assert not (root / "downloads" / "big.iso").exists()
    assert len(list(trash.glob("*"))) == 3


def test_delete_skips_already_gone_files(tmp_path):
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    (root / "temp" / "a.tmp").unlink()  # 预览之后、删除之前，文件已经没了
    report = delete_preview(preview, ["temp"], trash_dir=tmp_path / "trash")
    assert report["deleted"] == 0
    assert len(report["skipped"]) == 1


def test_delete_refuses_paths_outside_preview_root(tmp_path):
    root = build_tree(tmp_path / "tree")
    outside = tmp_path / "outside.txt"
    outside.write_text("别动我", encoding="utf-8")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    # 篡改预览，把根目录外的文件塞进已确认的种类里。
    preview["categories"]["temp"]["files"].append(str(outside))
    preview["categories"]["temp"]["file_count"] += 1
    report = delete_preview(preview, ["temp"], trash_dir=tmp_path / "trash")
    assert outside.exists()  # 根目录外的文件绝不能删
    assert len(report["failed"]) == 1


def test_format_summary_plain_language(tmp_path):
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    summary = format_summary(preview)
    assert "临时文件" in summary
    assert "日志文件" in summary
    assert "拿不准" in summary  # 低置信度的种类要标出来
    assert ".env" not in summary  # 密钥文件不出现在汇报里
    assert "key.pem" not in summary


# ---------- 多场景补充（BOOS 2026-11-10 要求：不只测正常路） ----------


def test_per_category_confirmation_only_confirmed_deleted(tmp_path):
    # 按种类确认：只点了 logs，temp 的东西一动不动。
    root = build_tree(tmp_path / "tree")
    preview = build_preview(scan(root, root_kind="other", large_threshold_mb=10))
    report = delete_preview(preview, ["logs"], trash_dir=tmp_path / "trash")
    assert report["deleted"] == 1  # 只有 app.log
    assert not (root / "logs" / "app.log").exists()
    assert (root / "temp" / "a.tmp").exists()
    assert (root / "downloads" / "setup.msi").exists()
    assert report["categories"]["logs"]["deleted"] == 1
    assert "temp" not in report["categories"]
