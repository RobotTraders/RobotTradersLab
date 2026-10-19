from importlib.resources import files
from pathlib import Path

from robottraderslab.scaffold.agents import lay_out_agent_files, main

_SHIPPED = files("robottraderslab") / "agents"
_INSTRUCTIONS = Path("agents", "AGENTS.md")
_SKILL_DIR = Path("agents", "skills", "writing-strategies")


def test_the_instructions_land_in_the_agents_folder(tmp_path):
    lay_out_agent_files(tmp_path)

    assert (tmp_path / _INSTRUCTIONS).read_text(encoding="utf-8") == (
        _SHIPPED / "AGENTS.md"
    ).read_text(encoding="utf-8")


def test_the_strategy_skill_lands_in_its_folder(tmp_path):
    lay_out_agent_files(tmp_path)

    assert (tmp_path / _SKILL_DIR / "SKILL.md").read_text(encoding="utf-8") == (
        _SHIPPED / "skills" / "writing-strategies" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_a_copy_already_there_is_written_afresh(tmp_path):
    stale = tmp_path / _SKILL_DIR / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    lay_out_agent_files(tmp_path)

    assert stale.read_text(encoding="utf-8") == (
        _SHIPPED / "skills" / "writing-strategies" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_main_names_each_file_it_wrote(tmp_path, capsys):
    main(tmp_path)

    assert set(capsys.readouterr().out.splitlines()) == {
        f"Wrote {tmp_path / _INSTRUCTIONS}",
        f"Wrote {tmp_path / _SKILL_DIR / 'SKILL.md'}",
    }
