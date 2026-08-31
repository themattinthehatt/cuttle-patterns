"""Tests for cuttle_patterns.cli.cmd_serve."""

import argparse
from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_serve import cmd_serve


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        port=5006,
        no_show=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_run_server_factory():
    calls = []

    def _fake_run_server(results_dir, port, show):
        calls.append({'results_dir': results_dir, 'port': port, 'show': show})

    _fake_run_server.calls = calls
    return _fake_run_server


class TestCmdServe:
    """Test the function cmd_serve."""

    def test_cmd_serve_missing_models_dir(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_serve(args)
        assert exc_info.value.code == 1
        assert 'no models found at' in capsys.readouterr().out

    def test_cmd_serve_launches_with_resolved_results_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        (results_dir / 'beast_models').mkdir(parents=True)
        fake_run_server = _fake_run_server_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_serve.run_server', fake_run_server)
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_serve(args)

        # Assert
        assert fake_run_server.calls == [
            {'results_dir': results_dir, 'port': 5006, 'show': True},
        ]

    def test_cmd_serve_passes_port_and_no_show(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        (results_dir / 'beast_models').mkdir(parents=True)
        fake_run_server = _fake_run_server_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_serve.run_server', fake_run_server)
        args = _make_args(results_dir=results_dir, port=8080, no_show=True)

        # Act
        cmd_serve(args)

        # Assert
        assert fake_run_server.calls == [
            {'results_dir': results_dir, 'port': 8080, 'show': False},
        ]
