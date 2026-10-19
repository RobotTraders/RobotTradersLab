from importlib.resources import files
from pathlib import Path

from robottraderslab.scaffold.claude import install_for_claude, main

_SHIPPED = files("robottraderslab") / "agents"
_SKILL = Path(".claude", "skills", "writing-strategies", "SKILL.md")
_RULE = Path(".claude", "rules", "robottraderslab.md")


def test_the_instructions_become_a_rule_of_the_project(tmp_path):
    install_for_claude(tmp_path)

    assert (tmp_path / _RULE).read_text(encoding="utf-8") == (
        _SHIPPED / "AGENTS.md"
    ).read_text(encoding="utf-8")


def test_each_shipped_skill_lands_under_the_claude_skills(tmp_path):
    install_for_claude(tmp_path)

    assert (tmp_path / _SKILL).read_text(encoding="utf-8") == (
        _SHIPPED / "skills" / "writing-strategies" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_the_project_instructions_are_untouched(tmp_path):
    own = tmp_path / "CLAUDE.md"
    own.write_text("mine", encoding="utf-8")

    install_for_claude(tmp_path)

    assert own.read_text(encoding="utf-8") == "mine"


def test_main_names_each_file_it_wrote(tmp_path, capsys):
    main(tmp_path)

    assert set(capsys.readouterr().out.splitlines()) == {
        f"Wrote {tmp_path / _RULE}",
        f"Wrote {tmp_path / _SKILL}",
    }
