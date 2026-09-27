import pytest

from robottraderslab.scaffold.examples import (
    ExampleError,
    copy_example,
    list_examples,
)


class TestListExamples:
    def test_returns_registered_names_sorted(self, register_example):
        register_example("momentum_demo", "fake_momentum_plugin")
        register_example("breakout_demo", "fake_breakout_plugin")

        names = list_examples()

        assert names == ["breakout_demo", "momentum_demo"]

    def test_with_no_examples_installed(self, monkeypatch):
        monkeypatch.setattr(
            "robottraderslab.scaffold.examples.entry_points", lambda group: []
        )

        names = list_examples()

        assert names == []


class TestCopyExample:
    def test_copies_example_files_into_suffixed_folder(
        self, register_example, tmp_path
    ):
        examples_dir = register_example("momentum_demo", "fake_momentum_plugin")
        (examples_dir / "momentum-bot-example.toml").write_text('strategy = "momentum"')
        workspace = tmp_path / "workspace"

        destination = copy_example("momentum_demo", workspace)

        assert destination == workspace / "momentum_demo-bot-example"
        assert (
            destination / "momentum-bot-example.toml"
        ).read_text() == 'strategy = "momentum"'

    def test_copies_nested_example_files(self, register_example, tmp_path):
        examples_dir = register_example("momentum_demo", "fake_momentum_plugin")
        (examples_dir / "data").mkdir()
        (examples_dir / "data" / "candles.csv").write_text("ts,open,close")
        workspace = tmp_path / "workspace"

        destination = copy_example("momentum_demo", workspace)

        assert (destination / "data" / "candles.csv").read_text() == "ts,open,close"

    def test_marks_the_destination_as_the_workspace_root(
        self, register_example, tmp_path
    ):
        register_example("momentum_demo", "fake_momentum_plugin")
        workspace = tmp_path / "workspace"

        copy_example("momentum_demo", workspace)

        assert (workspace / ".rtlab").is_dir()

    def test_a_second_example_joins_the_marked_workspace(
        self, register_example, tmp_path
    ):
        register_example("momentum_demo", "fake_momentum_plugin")
        register_example("breakout_demo", "fake_breakout_plugin")
        workspace = tmp_path / "workspace"
        copy_example("momentum_demo", workspace)
        kept = workspace / ".rtlab" / "kept.json"
        kept.write_text("{}")

        copy_example("breakout_demo", workspace)

        assert kept.read_text() == "{}"

    def test_with_unknown_name(self, register_example, tmp_path):
        register_example("momentum_demo", "fake_momentum_plugin")

        with pytest.raises(ExampleError, match="momentum_demo"):
            copy_example("unknown_demo", tmp_path / "workspace")

    def test_with_existing_destination(self, register_example, tmp_path):
        register_example("momentum_demo", "fake_momentum_plugin")
        (tmp_path / "workspace" / "momentum_demo-bot-example").mkdir(parents=True)

        with pytest.raises(ExampleError, match="already exists"):
            copy_example("momentum_demo", tmp_path / "workspace")

    def test_with_package_shipping_no_example_files(self, register_example, tmp_path):
        register_example("bare_demo", "fake_bare_plugin", with_examples=False)

        with pytest.raises(ExampleError, match="no example files"):
            copy_example("bare_demo", tmp_path / "workspace")
