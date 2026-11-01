import hashlib
import json
from importlib.resources import files
from pathlib import Path

import pytest

from robottraderslab.scaffold import EditedCopiesError
from robottraderslab.scaffold.agents import lay_out_agent_files, main

_SHIPPED = files("robottraderslab") / "agents"
_INSTRUCTIONS = Path("agents", "AGENTS.md")
_SKILL_DIR = Path("agents", "skills", "implementing-strategies")
_UPGRADE_SKILL = Path("agents", "skills", "rtlab-upgrade", "SKILL.md")
_MANIFEST = Path("agents", ".robottraderslab.json")
_ADVICE = (
    "An edit of yours belongs in your own instructions (CLAUDE.md, or the file "
    "your agent reads) or in a skill of your own; "
    "rerun with --force to replace these copies."
)


def test_the_instructions_land_in_the_agents_folder(tmp_path):
    lay_out_agent_files(tmp_path, force=False)

    assert (tmp_path / _INSTRUCTIONS).read_text(encoding="utf-8") == (
        _SHIPPED / "AGENTS.md"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("skill", ["implementing-strategies", "rtlab-upgrade"])
def test_each_shipped_skill_lands_in_its_folder(tmp_path, skill):
    lay_out_agent_files(tmp_path, force=False)

    assert (tmp_path / "agents" / "skills" / skill / "SKILL.md").read_text(
        encoding="utf-8"
    ) == (_SHIPPED / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")


def test_a_copy_already_there_is_written_afresh(tmp_path):
    stale = tmp_path / _SKILL_DIR / "SKILL.md"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    lay_out_agent_files(tmp_path, force=False)

    assert stale.read_text(encoding="utf-8") == (
        _SHIPPED / "skills" / "implementing-strategies" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_main_names_each_file_it_wrote(tmp_path, capsys):
    main(tmp_path, force=False)

    assert set(capsys.readouterr().out.splitlines()) == {
        f"Wrote {tmp_path / _INSTRUCTIONS}",
        f"Wrote {tmp_path / _SKILL_DIR / 'SKILL.md'}",
        f"Wrote {tmp_path / _UPGRADE_SKILL}",
    }


def test_the_manifest_records_the_hash_of_each_file_written(tmp_path):
    lay_out_agent_files(tmp_path, force=False)

    recorded = json.loads((tmp_path / _MANIFEST).read_text(encoding="utf-8"))
    assert set(recorded) == {
        "agents/AGENTS.md",
        "agents/skills/implementing-strategies/SKILL.md",
        "agents/skills/rtlab-upgrade/SKILL.md",
    }
    assert (
        recorded["agents/AGENTS.md"]
        == hashlib.sha256((_SHIPPED / "AGENTS.md").read_bytes()).hexdigest()
    )


def test_an_unedited_copy_is_not_refused_on_a_rerun(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    instructions = tmp_path / _INSTRUCTIONS
    shipped = instructions.read_bytes()

    lay_out_agent_files(tmp_path, force=False)

    assert instructions.read_bytes() == shipped


def test_a_copy_checked_out_with_crlf_endings_is_not_refused(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    instructions = tmp_path / _INSTRUCTIONS
    instructions.write_bytes(instructions.read_bytes().replace(b"\n", b"\r\n"))

    lay_out_agent_files(tmp_path, force=False)

    assert instructions.read_bytes() == (_SHIPPED / "AGENTS.md").read_bytes()


def test_an_edited_copy_is_refused_and_named(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    edited = tmp_path / _INSTRUCTIONS
    edited.write_text("mine", encoding="utf-8")
    also_edited = tmp_path / _SKILL_DIR / "SKILL.md"
    also_edited.write_text("also mine", encoding="utf-8")

    with pytest.raises(EditedCopiesError) as refused:
        lay_out_agent_files(tmp_path, force=False)

    assert str(refused.value).splitlines() == [
        f"Edited since rtlab init wrote it: {edited}",
        f"Edited since rtlab init wrote it: {also_edited}",
        _ADVICE,
    ]


def test_a_refusal_writes_nothing(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    edited = tmp_path / _INSTRUCTIONS
    edited.write_text("mine", encoding="utf-8")
    deleted = tmp_path / _UPGRADE_SKILL
    deleted.unlink()
    manifest = (tmp_path / _MANIFEST).read_text(encoding="utf-8")

    with pytest.raises(EditedCopiesError):
        lay_out_agent_files(tmp_path, force=False)

    assert edited.read_text(encoding="utf-8") == "mine"
    assert not deleted.exists()
    assert (tmp_path / _MANIFEST).read_text(encoding="utf-8") == manifest


def test_force_overwrites_an_edited_copy(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    edited = tmp_path / _INSTRUCTIONS
    edited.write_text("mine", encoding="utf-8")

    lay_out_agent_files(tmp_path, force=True)

    assert edited.read_text(encoding="utf-8") == (_SHIPPED / "AGENTS.md").read_text(
        encoding="utf-8"
    )


def test_a_copy_without_a_manifest_entry_is_overwritten_and_recorded(tmp_path):
    predating = tmp_path / _INSTRUCTIONS
    predating.parent.mkdir(parents=True)
    predating.write_text("written by an earlier version", encoding="utf-8")

    lay_out_agent_files(tmp_path, force=False)

    recorded = json.loads((tmp_path / _MANIFEST).read_text(encoding="utf-8"))
    assert (
        recorded[_INSTRUCTIONS.as_posix()]
        == hashlib.sha256(predating.read_bytes()).hexdigest()
    )


def test_a_recorded_skill_the_engine_no_longer_ships_is_removed(tmp_path, capsys):
    lay_out_agent_files(tmp_path, force=False)
    retired = tmp_path / "agents" / "skills" / "retired" / "SKILL.md"
    retired.parent.mkdir()
    retired.write_text("old", encoding="utf-8")
    manifest = tmp_path / _MANIFEST
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    recorded["agents/skills/retired/SKILL.md"] = hashlib.sha256(b"old").hexdigest()
    manifest.write_text(json.dumps(recorded), encoding="utf-8")
    capsys.readouterr()

    main(tmp_path, force=False)

    assert not retired.parent.exists()
    assert f"Removed {retired}" in capsys.readouterr().out.splitlines()


def test_a_skill_folder_the_manifest_does_not_record_is_left_alone(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    own = tmp_path / "agents" / "skills" / "mine" / "SKILL.md"
    own.parent.mkdir()
    own.write_text("mine", encoding="utf-8")

    lay_out_agent_files(tmp_path, force=False)

    assert own.read_text(encoding="utf-8") == "mine"


def _retire_skill(project, content):
    retired = project / "agents" / "skills" / "retired" / "SKILL.md"
    retired.parent.mkdir()
    retired.write_text(content, encoding="utf-8")
    manifest = project / _MANIFEST
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    recorded["agents/skills/retired/SKILL.md"] = hashlib.sha256(b"old").hexdigest()
    manifest.write_text(json.dumps(recorded), encoding="utf-8")
    return retired


def test_an_edited_file_the_engine_no_longer_ships_is_refused_not_deleted(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    retired = _retire_skill(tmp_path, "edited")

    with pytest.raises(EditedCopiesError, match="retired"):
        lay_out_agent_files(tmp_path, force=False)

    assert retired.read_text(encoding="utf-8") == "edited"


def test_force_deletes_an_edited_file_the_engine_no_longer_ships(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    retired = _retire_skill(tmp_path, "edited")

    lay_out_agent_files(tmp_path, force=True)

    assert not retired.exists()


def test_a_retired_skill_folder_holding_a_file_of_the_users_is_kept(tmp_path):
    lay_out_agent_files(tmp_path, force=False)
    retired = _retire_skill(tmp_path, "old")
    own = retired.parent / "notes.md"
    own.write_text("mine", encoding="utf-8")

    lay_out_agent_files(tmp_path, force=False)

    assert own.read_text(encoding="utf-8") == "mine"
    assert not retired.exists()


def test_a_corrupt_manifest_is_refused_naming_the_file(tmp_path):
    manifest = tmp_path / _MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{not json", encoding="utf-8")

    with pytest.raises(EditedCopiesError, match=r"\.robottraderslab\.json.*delete"):
        lay_out_agent_files(tmp_path, force=False)

    assert manifest.read_text(encoding="utf-8") == "{not json"


def test_a_manifest_that_is_not_utf8_is_refused_naming_the_file(tmp_path):
    manifest = tmp_path / _MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"\xff\xfe{}")

    with pytest.raises(EditedCopiesError, match=r"\.robottraderslab\.json.*UTF-8"):
        lay_out_agent_files(tmp_path, force=False)


@pytest.mark.parametrize(
    "content", ["[1, 2]", '{"agents/AGENTS.md": 3}'], ids=["list", "non-string-hash"]
)
def test_a_manifest_not_mapping_paths_to_hashes_is_refused(tmp_path, content):
    manifest = tmp_path / _MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(content, encoding="utf-8")

    with pytest.raises(EditedCopiesError, match="does not map paths to hashes"):
        lay_out_agent_files(tmp_path, force=False)


def test_a_manifest_key_outside_the_project_is_refused_and_never_deleted(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside.txt"
    outside.write_text("kept", encoding="utf-8")
    manifest = project / _MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"../outside.txt": "0" * 64}), encoding="utf-8")

    with pytest.raises(EditedCopiesError, match="outside the project"):
        lay_out_agent_files(project, force=True)

    assert outside.read_text(encoding="utf-8") == "kept"
