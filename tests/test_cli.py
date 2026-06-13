import pytest

from modelinfo import __version__
from modelinfo.cli import parse_args


def test_version_flag_prints_installed_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--version"])

    assert exc_info.value.code == 0
    assert f"modelinfo {__version__}" in capsys.readouterr().out
