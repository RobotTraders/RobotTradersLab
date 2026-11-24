from robottraderslab.scaffold import project_folder


def test_the_nearest_project_wins(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    inner = tmp_path / "plugins" / "one"
    (inner / "src").mkdir(parents=True)
    (inner / "pyproject.toml").touch()

    assert project_folder(inner / "src") == inner
