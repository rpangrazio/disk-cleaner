import os
import tempfile
import shutil
import json
import subprocess


def run_cmd(args, cwd=None):
    p = subprocess.run(["python", "disk_cleaner.py"] + args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout + "\n" + p.stderr


def test_md5_duplicates_and_plan(tmp_path):
    # Create sample files
    src = tmp_path / "src"
    src.mkdir()
    f1 = src / "a.txt"
    f2 = src / "a_copy.txt"
    f3 = src / "b.txt"
    f1.write_text("hello world")
    f2.write_text("hello world")
    f3.write_text("different content")

    report = tmp_path / "report.json"

    code, out = run_cmd(["--source", str(src), "--report", str(report), "--md5-all"])  # force md5 for all
    assert code == 0, out
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["summary"]["total_files"] == 3
    # there should be at least one duplicate group by hash
    assert data["summary"]["duplicate_hash_groups"] >= 1


def test_apply_copy_plan(tmp_path):
    src = tmp_path / "src2"
    src.mkdir()
    (src / "x.jpg").write_text("binarycontent", encoding="utf-8")
    (src / "y.png").write_text("morebinary", encoding="utf-8")
    dest = tmp_path / "out"
    report = tmp_path / "r2.json"
    code, out = run_cmd(["--source", str(src), "--dest", str(dest), "--report", str(report)])
    assert code == 0, out
