import json
from importlib.resources import files
from pathlib import Path

import pytest

from robottraderslab.scaffold.claude import install_for_claude, main

_SHIPPED = files("robottraderslab") / "agents"
_SKILL = Path(".claude", "skills", "implementing-strategies", "SKILL.md")
_UPGRADE_SKILL = Path(".claude", "skills", "rtlab-upgrade", "SKILL.md")
_RULE = Path(".claude", "rules", "robottraderslab.md")


def test_the_instructions_become_a_rule_of_the_project(tmp_path):
    install_for_claude(tmp_path, force=False)

    assert (tmp_path / _RULE).read_text(encoding="utf-8") == (
        _SHIPPED / "AGENTS.md"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("skill", ["implementing-strategies", "rtlab-upgrade"])
def test_each_shipped_skill_lands_under_the_claude_skills(tmp_path, skill):
    install_for_claude(tmp_path, force=False)

    assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").read_text(
        encoding="utf-8"
    ) == (_SHIPPED / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")


def test_the_project_instructions_are_untouched(tmp_path):
    own = tmp_path / "CLAUDE.md"
    own.write_text("mine", encoding="utf-8")

    install_for_claude(tmp_path, force=False)

    assert own.read_text(encoding="utf-8") == "mine"


def test_main_names_each_file_it_wrote(tmp_path, capsys):
    main(tmp_path, force=False)

    assert set(capsys.readouterr().out.splitlines()) == {
        f"Wrote {tmp_path / _RULE}",
        f"Wrote {tmp_path / _SKILL}",
        f"Wrote {tmp_path / _UPGRADE_SKILL}",
    }


def test_the_manifest_sits_beside_the_skills_and_records_the_rule(tmp_path):
    install_for_claude(tmp_path, force=False)

    recorded = json.loads(
        (tmp_path / ".claude" / "skills" / ".robottraderslab.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(recorded) == {
        ".claude/rules/robottraderslab.md",
        ".claude/skills/implementing-strategies/SKILL.md",
        ".claude/skills/rtlab-upgrade/SKILL.md",
    }
