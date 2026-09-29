import os
import stat
import sys
from pathlib import Path

import pytest

from robottraderslab.cli import app

FAKE_CRONTAB = """
import os
import sys
from pathlib import Path

stored = Path(os.environ["FAKE_CRONTAB_FILE"])
refusal = os.environ.get("FAKE_CRONTAB_REFUSAL")
if sys.argv[1] == "-" and os.environ.get("FAKE_CRONTAB_WRITE_REFUSAL"):
    refusal = os.environ["FAKE_CRONTAB_WRITE_REFUSAL"]
if refusal:
    sys.stderr.write(refusal + "\\n")
    sys.exit(1)
if sys.argv[1] == "-l":
    if not stored.exists():
        sys.stderr.write("no crontab for tester\\n")
        sys.exit(1)
    sys.stdout.write(stored.read_text())
elif sys.argv[1] == "-":
    stored.write_text(sys.stdin.read())
"""

BACKUP_JOB = "0 3 * * * /usr/local/bin/backup --all"
COMMENT = "# nightly backup of the home folder"
MAIL = "MAILTO=tester@example.invalid"


@pytest.fixture
def uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    path = (tmp_path / "home" / ".local" / "bin" / "uv").as_posix()
    monkeypatch.setenv("UV", path)
    return path


@pytest.fixture
def crontab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_crontab.py"
    script.write_text(FAKE_CRONTAB)
    if os.name == "nt":
        (bin_dir / "crontab.bat").write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        command = bin_dir / "crontab"
        command.write_text(f"#!{sys.executable}\n{FAKE_CRONTAB}")
        command.chmod(command.stat().st_mode | stat.S_IEXEC)
    stored = tmp_path / "crontab.txt"
    monkeypatch.setenv("FAKE_CRONTAB_FILE", str(stored))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return stored


def scheduler_line(uv: str, project: Path, registry: str) -> str:
    return (
        f"* * * * * {uv} run --directory {project.as_posix()} rtlab scheduler "
        f"{registry}"
    )


def write_crontab(stored: Path, *lines: str) -> None:
    stored.write_text("".join(f"{line}\n" for line in lines))


def read_crontab(stored: Path) -> list[str]:
    return stored.read_text().splitlines()


def test_a_user_with_no_crontab(cli, project, uv, crontab):
    installed = cli.invoke(app, ["init", "cron"])

    expected = scheduler_line(uv, project, "workspace/registry.toml")
    assert installed.exit_code == 0
    assert read_crontab(crontab) == [expected]
    assert installed.output == f"Added to the crontab:\n{expected}\n"


def test_the_line_is_appended_below_the_other_entries(cli, project, uv, crontab):
    write_crontab(crontab, MAIL, COMMENT, BACKUP_JOB)

    cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [
        MAIL,
        COMMENT,
        BACKUP_JOB,
        scheduler_line(uv, project, "workspace/registry.toml"),
    ]


def test_the_line_already_there_leaves_the_crontab_byte_for_byte(
    cli, project, uv, crontab
):
    write_crontab(
        crontab,
        COMMENT,
        BACKUP_JOB,
        scheduler_line(uv, project, "workspace/registry.toml"),
    )
    before = crontab.read_bytes()

    kept = cli.invoke(app, ["init", "cron"])

    assert kept.exit_code == 0
    assert crontab.read_bytes() == before
    assert kept.output.startswith("The crontab already holds the scheduler line:")


def test_a_second_run_holds_the_line_once(cli, project, uv, crontab):
    write_crontab(crontab, COMMENT, BACKUP_JOB)

    cli.invoke(app, ["init", "cron"])
    cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [
        COMMENT,
        BACKUP_JOB,
        scheduler_line(uv, project, "workspace/registry.toml"),
    ]


OLDER_SPELLINGS = [
    "* * * * * cd {project} && /root/.local/bin/uv run python -m robottraderslab "
    "scheduler workspace/registry.toml >> workspace/cron.log 2>&1",
    "* * * * * /old/uv run --directory {project} rtlab scheduler "
    "{project}/workspace/registry.toml",
    "*/1 * * * * /old/uv run --directory={project}/ robottraderslab scheduler "
    "./workspace/registry.toml",
    "* * * * * cd {project} && uv run rtlab scheduler workspace/registry.toml",
]


@pytest.mark.parametrize(
    "older",
    OLDER_SPELLINGS,
    ids=["cd and module", "absolute registry", "directory equals", "cd and script"],
)
def test_an_older_line_for_the_registry_is_replaced_where_it_stands(
    older, cli, project, uv, crontab
):
    older_line = older.format(project=project.as_posix())
    write_crontab(crontab, COMMENT, older_line, BACKUP_JOB)

    replaced = cli.invoke(app, ["init", "cron"])

    expected = scheduler_line(uv, project, "workspace/registry.toml")
    assert read_crontab(crontab) == [COMMENT, expected, BACKUP_JOB]
    assert (
        replaced.output == f"Replaced in the crontab:\n{older_line}\nby:\n{expected}\n"
    )


HOME_SPELLINGS = [
    "* * * * * cd ~/RobotTradersLab && uv run rtlab scheduler workspace/registry.toml",
    "* * * * * cd RobotTradersLab && .venv/bin/rtlab scheduler workspace/registry.toml",
]


@pytest.mark.parametrize(
    "older", HOME_SPELLINGS, ids=["cd to home", "cd from home and venv script"]
)
def test_a_line_naming_the_project_from_the_home_folder_is_replaced(
    older, cli, project, uv, crontab, monkeypatch
):
    monkeypatch.setenv("HOME", project.parent.as_posix())
    monkeypatch.setenv("USERPROFILE", str(project.parent))
    write_crontab(crontab, older, BACKUP_JOB)

    cli.invoke(app, ["init", "cron"])

    expected = scheduler_line(uv, project, "workspace/registry.toml")
    assert read_crontab(crontab) == [expected, BACKUP_JOB]


def test_a_duplicate_line_for_the_registry_is_removed(cli, project, uv, crontab):
    line = scheduler_line(uv, project, "workspace/registry.toml")
    write_crontab(crontab, line, BACKUP_JOB, line)

    cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [line, BACKUP_JOB]


def test_another_registry_of_the_project_is_left_in_place(cli, project, uv, crontab):
    pilot = scheduler_line("/old/uv", project, "workspace/pilot-registry.toml")
    write_crontab(crontab, pilot)

    added = cli.invoke(app, ["init", "cron"])

    expected = scheduler_line(uv, project, "workspace/registry.toml")
    assert read_crontab(crontab) == [pilot, expected]
    assert added.output.endswith(f"Other scheduler lines left in place:\n{pilot}\n")


def test_the_registry_option_updates_only_that_registry(cli, project, uv, crontab):
    main_line = scheduler_line(uv, project, "workspace/registry.toml")
    pilot = scheduler_line("/old/uv", project, "workspace/pilot-registry.toml")
    write_crontab(crontab, main_line, pilot)

    cli.invoke(app, ["init", "cron", "--registry", "workspace/pilot-registry.toml"])

    assert read_crontab(crontab) == [
        main_line,
        scheduler_line(uv, project, "workspace/pilot-registry.toml"),
    ]


def test_the_registry_option_names_the_registry_the_line_ends_on(
    cli, project, uv, crontab
):
    cli.invoke(app, ["init", "cron", "--registry", "workspace/pilot-registry.toml"])

    assert read_crontab(crontab) == [
        scheduler_line(uv, project, "workspace/pilot-registry.toml")
    ]


def test_another_project_folder_is_left_in_place(cli, project, uv, crontab):
    other = "* * * * * /root/.local/bin/uv run --directory /root/OtherLab rtlab scheduler workspace/registry.toml"
    write_crontab(crontab, other)

    added = cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [
        other,
        scheduler_line(uv, project, "workspace/registry.toml"),
    ]
    assert added.output.endswith(f"Other scheduler lines left in place:\n{other}\n")


def test_run_from_a_folder_inside_the_project(cli, project, uv, crontab, monkeypatch):
    monkeypatch.chdir(project / "workspace")

    cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [
        scheduler_line(uv, project, "workspace/registry.toml")
    ]


def test_a_folder_outside_any_project(cli, tmp_path, uv, crontab, monkeypatch):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)

    refused = cli.invoke(app, ["init", "cron"])

    assert refused.exit_code == 1
    assert "no `pyproject.toml` at or above" in refused.stderr
    assert not crontab.exists()


def test_a_machine_without_cron(cli, project, uv, monkeypatch, tmp_path):
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    refused = cli.invoke(app, ["init", "cron"])

    assert refused.exit_code == 1
    assert "`crontab` is not on PATH" in refused.stderr


def test_a_commented_scheduler_line_stays_as_written(cli, project, uv, crontab):
    commented = "# " + scheduler_line("/old/uv", project, "workspace/registry.toml")
    write_crontab(crontab, commented)

    added = cli.invoke(app, ["init", "cron"])

    expected = scheduler_line(uv, project, "workspace/registry.toml")
    assert read_crontab(crontab) == [commented, expected]
    assert added.output == f"Added to the crontab:\n{expected}\n"


def test_a_crontab_that_cannot_be_read_is_left_alone(
    cli, project, uv, crontab, monkeypatch
):
    write_crontab(crontab, BACKUP_JOB)
    before = crontab.read_bytes()
    monkeypatch.setenv("FAKE_CRONTAB_REFUSAL", "crontab: permission denied")

    refused = cli.invoke(app, ["init", "cron"])

    assert refused.exit_code == 1
    assert "cannot read the crontab: crontab: permission denied" in refused.stderr
    assert crontab.read_bytes() == before


def test_uv_found_on_path_when_not_run_through_uv(
    cli, project, crontab, monkeypatch, tmp_path
):
    monkeypatch.delenv("UV", raising=False)
    uv_dir = tmp_path / "uv-bin"
    uv_dir.mkdir()
    uv_command = uv_dir / ("uv.EXE" if os.name == "nt" else "uv")
    uv_command.write_text("")
    uv_command.chmod(uv_command.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{uv_dir}{os.pathsep}{os.environ['PATH']}")

    cli.invoke(app, ["init", "cron"])

    assert read_crontab(crontab) == [
        scheduler_line(uv_command.as_posix(), project, "workspace/registry.toml")
    ]


def test_a_crontab_that_cannot_be_written(cli, project, uv, crontab, monkeypatch):
    write_crontab(crontab, BACKUP_JOB)
    monkeypatch.setenv("FAKE_CRONTAB_WRITE_REFUSAL", "crontab: disk full")

    refused = cli.invoke(app, ["init", "cron"])

    assert refused.exit_code == 1
    assert "cannot write the crontab: crontab: disk full" in refused.stderr
