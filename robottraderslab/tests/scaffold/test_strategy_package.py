import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

from robottraderslab.scaffold import StrategyPackageError
from robottraderslab.scaffold.strategy_package import (
    create_strategy_package,
    install_package,
    main,
)


@pytest.fixture
def pyproject(package_folder: Path, read_pyproject) -> dict:
    return read_pyproject(package_folder)


@pytest.fixture
def bot(package_folder: Path, workspace: Path) -> dict:
    return tomllib.loads((workspace / "my-strategy" / "bot.toml").read_text("utf-8"))


class TestCreateStrategyPackage:
    def test_writes_the_package_files_and_the_bot_and_names_them(
        self, tmp_path, workspace
    ):
        folder = tmp_path / "my-strategy"

        written = create_strategy_package(folder, workspace)

        assert written == [
            folder / "pyproject.toml",
            folder / ".gitignore",
            folder / "src" / "my_strategy" / "__init__.py",
            folder / "src" / "my_strategy" / "main.py",
            folder / "src" / "my_strategy" / "chart_indicators.py",
            workspace / "my-strategy" / "bot.toml",
        ]
        assert all(path.is_file() for path in written)

    def test_the_package_init_is_empty(self, package_folder):
        init = package_folder / "src" / "my_strategy" / "__init__.py"

        assert init.read_text(encoding="utf-8") == ""

    def test_the_package_gitignore_keeps_the_caches_and_the_environment_out(
        self, package_folder
    ):
        ignored = (package_folder / ".gitignore").read_text(encoding="utf-8")

        assert ignored.splitlines() == ["__pycache__/", ".venv/"]

    def test_the_workspace_is_marked_as_the_root(self, package_folder, workspace):
        assert (workspace / ".rtlab").is_dir()
        assert (workspace / ".gitignore").is_file()

    def test_pyproject_registers_the_class_of_main_and_the_chart_lines_under_the_module_name(
        self, pyproject
    ):
        assert pyproject["project"]["name"] == "my-strategy"
        assert pyproject["project"]["entry-points"] == {
            "robot_traders_lab.strategies": {
                "my_strategy": "my_strategy.main:MyStrategy"
            },
            "robot_traders_lab.indicators": {
                "my_strategy": "my_strategy.chart_indicators:chart_lines"
            },
        }

    def test_pyproject_depends_on_the_installed_engine_version(self, pyproject):
        assert pyproject["project"]["dependencies"] == [
            f"robottraderslab>={version('robottraderslab')}"
        ]

    def test_the_bot_names_the_strategy_by_its_entry_point(self, bot, pyproject):
        registered = pyproject["project"]["entry-points"][
            "robot_traders_lab.strategies"
        ]

        assert [bot["strategy"]["strategy_class"]] == list(registered)

    def test_the_bot_backtests_a_hyperliquid_perpetual_at_its_base_fees(
        self, bot, btc_4h_profile
    ):
        assert bot["strategy"]["profiles"] == [btc_4h_profile]
        assert bot["backtest"]["initial_balance"] == {"USDC": 10000.0}
        assert bot["backtest"]["maker_fee_rate"] == 0.00015
        assert bot["backtest"]["taker_fee_rate"] == 0.00045
        assert bot["backtest"]["ohlcv_provider"] == {
            "ohlcv_provider": "ccxt_hyperliquid"
        }

    @pytest.mark.parametrize(
        ("folder_name", "module", "strategy_class", "profile_class"),
        [
            ("my-strategy", "my_strategy", "MyStrategy", "MyStrategyProfile"),
            ("momentum", "momentum", "MomentumStrategy", "MomentumStrategyProfile"),
            ("ma_cross2", "ma_cross2", "MaCross2Strategy", "MaCross2StrategyProfile"),
        ],
    )
    def test_names_follow_the_folder_name(
        self,
        folder_name,
        module,
        strategy_class,
        profile_class,
        tmp_path,
        workspace,
        read_pyproject,
    ):
        folder = tmp_path / folder_name

        create_strategy_package(folder, workspace)

        source = (folder / "src" / module / "main.py").read_text(encoding="utf-8")
        assert f"class {profile_class}(Profile):" in source
        assert f"class {strategy_class}(ProfileStrategy[{profile_class}]):" in source
        pyproject = read_pyproject(folder)
        entry_points = pyproject["project"]["entry-points"]
        assert entry_points["robot_traders_lab.strategies"] == {
            module: f"{module}.main:{strategy_class}"
        }
        assert entry_points["robot_traders_lab.indicators"] == {
            module: f"{module}.chart_indicators:chart_lines"
        }
        assert (workspace / folder_name / "bot.toml").is_file()

    def test_an_existing_package_folder_is_not_overwritten(self, tmp_path, workspace):
        folder = tmp_path / "my-strategy"
        folder.mkdir()
        (folder / "mine.txt").write_text("kept", encoding="utf-8")

        with pytest.raises(StrategyPackageError, match="already exists"):
            create_strategy_package(folder, workspace)

        assert [path.name for path in folder.iterdir()] == ["mine.txt"]
        assert not workspace.exists()

    def test_an_existing_bot_folder_refuses_before_anything_is_created(
        self, tmp_path, workspace
    ):
        (workspace / "my-strategy").mkdir(parents=True)
        folder = tmp_path / "my-strategy"

        with pytest.raises(StrategyPackageError, match="already exists"):
            create_strategy_package(folder, workspace)

        assert not folder.exists()
        assert list((workspace / "my-strategy").iterdir()) == []

    @pytest.mark.parametrize(
        "folder_name",
        [
            "My-Strategy",
            "1strategy",
            "my strategy",
            "-strategy",
            "my--strategy",
            "class",
        ],
    )
    def test_a_name_that_cannot_be_a_python_package(
        self, folder_name, tmp_path, workspace
    ):
        folder = tmp_path / folder_name

        with pytest.raises(StrategyPackageError, match="cannot be a Python package"):
            create_strategy_package(folder, workspace)

        assert not folder.exists()
        assert not workspace.exists()

    @pytest.mark.parametrize("folder_name", ["json", "robottraderslab", "numpy"])
    def test_a_name_an_importable_module_already_holds(
        self, folder_name, tmp_path, workspace
    ):
        folder = tmp_path / folder_name

        with pytest.raises(StrategyPackageError, match="already an importable module"):
            create_strategy_package(folder, workspace)

        assert not folder.exists()
        assert not workspace.exists()

    def test_a_name_an_installed_package_registers_a_strategy_under(
        self, tmp_path, workspace
    ):
        folder = tmp_path / "futures-ma"

        with pytest.raises(StrategyPackageError, match="already registered"):
            create_strategy_package(folder, workspace)

        assert not folder.exists()
        assert not workspace.exists()

    def test_a_name_that_would_shadow_the_engines_class(self, tmp_path, workspace):
        folder = tmp_path / "profile-strategy"

        with pytest.raises(StrategyPackageError, match="`ProfileStrategy`"):
            create_strategy_package(folder, workspace)

        assert not folder.exists()
        assert not workspace.exists()


class TestInstallPackage:
    def test_adds_the_package_editable_from_the_project_folder(
        self, tmp_path, fake_uv, installs
    ):
        folder = tmp_path / "my-strategy"
        project = tmp_path / "project"

        install_package(folder, project)

        assert installs == [([fake_uv, "add", "--editable", str(folder)], project)]

    def test_a_failed_install_names_the_line_to_run_by_hand(
        self, tmp_path, failed_install
    ):
        folder = tmp_path / "my-strategy"
        project = tmp_path / "project"

        with pytest.raises(StrategyPackageError) as raised:
            install_package(folder, project)

        message = str(raised.value)
        assert "exit code 2" in message
        assert f"run `uv add --editable {folder.resolve()}` in '{project}'" in message

    def test_the_line_to_run_by_hand_quotes_a_path_holding_a_space(
        self, tmp_path, failed_install
    ):
        folder = tmp_path / "my projects" / "my-strategy"

        with pytest.raises(StrategyPackageError) as raised:
            install_package(folder, tmp_path)

        assert f'--editable "{folder.resolve()}"' in str(raised.value)

    def test_uv_missing_names_the_line_to_run_by_hand(self, tmp_path, monkeypatch):
        monkeypatch.delenv("UV", raising=False)
        monkeypatch.setenv("PATH", "")

        with pytest.raises(StrategyPackageError, match="`uv` is not on PATH; run `uv"):
            install_package(tmp_path / "my-strategy", tmp_path)

    def test_uv_that_cannot_start_names_the_line_to_run_by_hand(
        self, tmp_path, fake_uv, monkeypatch
    ):
        def refuse(command, cwd, **options):
            raise PermissionError("denied")

        monkeypatch.setattr("subprocess.run", refuse)

        with pytest.raises(StrategyPackageError, match="denied; run `uv add"):
            install_package(tmp_path / "my-strategy", tmp_path)


class TestMain:
    def test_creates_the_package_resolved_from_the_project_and_installs_it(
        self, tmp_path, fake_uv, installs, capsys
    ):
        project = tmp_path / "project"
        workspace = project / "workspace"

        main(Path("../my-strategy"), project, workspace)

        package = (tmp_path / "my-strategy").resolve()
        assert capsys.readouterr().out.splitlines() == [
            f"Wrote {package / 'pyproject.toml'}",
            f"Wrote {package / '.gitignore'}",
            f"Wrote {package / 'src' / 'my_strategy' / '__init__.py'}",
            f"Wrote {package / 'src' / 'my_strategy' / 'main.py'}",
            f"Wrote {package / 'src' / 'my_strategy' / 'chart_indicators.py'}",
            f"Wrote {workspace / 'my-strategy' / 'bot.toml'}",
            f"Installed my-strategy in {project}",
        ]
        assert installs == [([fake_uv, "add", "--editable", str(package)], project)]

    def test_a_failed_install_leaves_the_created_files(
        self, tmp_path, failed_install, capsys
    ):
        project = tmp_path / "project"

        with pytest.raises(StrategyPackageError, match="the package stays at"):
            main(Path("my-strategy"), project, project / "workspace")

        assert (project / "my-strategy" / "pyproject.toml").is_file()
        assert (project / "workspace" / "my-strategy" / "bot.toml").is_file()
        assert "Wrote" in capsys.readouterr().out

    def test_a_refusal_installs_nothing(self, tmp_path, installs):
        project = tmp_path / "project"
        (project / "my-strategy").mkdir(parents=True)

        with pytest.raises(StrategyPackageError, match="already exists"):
            main(Path("my-strategy"), project, project / "workspace")

        assert installs == []
