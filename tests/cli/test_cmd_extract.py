"""Tests for cuttle_patterns.cli.cmd_extract."""

import sys

import pytest

from cuttle_patterns.cli.cmd_extract import STAGES, cmd_extract, register
from cuttle_patterns.cli.main import main


class TestCmdExtract:
    """Test the function cmd_extract."""

    def test_cmd_extract_prints_every_stage_in_order(self, capsys: pytest.CaptureFixture):
        # Act
        cmd_extract(None)

        # Assert
        out = capsys.readouterr().out
        positions = [out.index(f'cuttle {name}') for name, _ in STAGES]
        assert positions == sorted(positions)
        assert len(STAGES) == 7

    def test_every_stage_is_a_registered_command(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(sys, 'argv', ['cuttle', '--help'])

        # Act
        with pytest.raises(SystemExit):
            main()

        # Assert
        out = capsys.readouterr().out
        assert all(name in out for name, _ in STAGES)

    def test_register_wires_the_handler(self, parse_cli):
        assert parse_cli(register, ['extract']).handler is cmd_extract

    def test_main_runs_extract_without_arguments(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr(sys, 'argv', ['cuttle', 'extract'])

        # Act
        main()

        # Assert
        assert 'extract-eligibility' in capsys.readouterr().out
