from pathlib import Path

import pytest

from robottraderslab.outdated import OutdatedError
from robottraderslab.outdated.moves import Move, parse_moves, project_moves

UV_DRY_RUN = """\
Resolved 112 packages in 41ms
Update pandas v2.3.1 -> v2.3.3
Update robottraderslab v0.8.0 -> v0.10.0
Update robottraderslab-envelope v0.3.1 -> v0.4.0
Add numpy v2.0.0
Update robottraderslab-bitget v1.2.0 -> v1.3.0
Remove robottraderslab-oanda v0.1.0
"""


def test_only_the_robottraderslab_updates_are_kept_in_the_order_uv_printed_them():
    parsed = parse_moves(UV_DRY_RUN)

    assert parsed == [
        Move("robottraderslab", "0.8.0", "0.10.0"),
        Move("robottraderslab-envelope", "0.3.1", "0.4.0"),
        Move("robottraderslab-bitget", "1.2.0", "1.3.0"),
    ]


def test_the_dry_run_runs_in_the_project_folder(
    locked_project, make_uv_answer, tmp_path, monkeypatch
):
    call = make_uv_answer(UV_DRY_RUN)
    monkeypatch.chdir(tmp_path)

    project_moves(locked_project)

    _, folder = call.read_text(encoding="utf-8").split("\n")
    assert Path(folder).samefile(locked_project)


def test_the_dry_run_asks_uv_for_the_upgrade_plan_without_writing_the_lock_or_colour(
    locked_project, make_uv_answer
):
    call = make_uv_answer("Resolved 112 packages in 41ms\n")

    project_moves(locked_project)

    arguments, _ = call.read_text(encoding="utf-8").split("\n")
    assert arguments == "lock --upgrade --dry-run --color never"


def test_a_failed_resolution_carries_what_uv_said(locked_project, make_uv_answer):
    make_uv_answer("error: No solution found when resolving dependencies\n", 2)

    with pytest.raises(OutdatedError, match="No solution found") as refused:
        project_moves(locked_project)

    assert str(locked_project) in str(refused.value)


def test_uv_that_cannot_be_started(locked_project, monkeypatch):
    monkeypatch.setenv("UV", str(locked_project / "no-such-folder" / "uv"))

    with pytest.raises(OutdatedError, match="cannot run `.*uv`"):
        project_moves(locked_project)


def test_uv_found_neither_by_variable_nor_on_path(locked_project, monkeypatch):
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setenv("PATH", "")

    with pytest.raises(OutdatedError, match="`uv` is not on PATH"):
        project_moves(locked_project)


def test_a_folder_holding_no_lock(project, make_uv_answer):
    call = make_uv_answer("Update robottraderslab v0.8.0 -> v0.10.0\n")

    with pytest.raises(OutdatedError, match="uv.lock") as refused:
        project_moves(project)

    assert str(project) in str(refused.value)
    assert not call.exists()
