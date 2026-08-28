"""Tests for cuttle_patterns.cli.cmd_predict."""

import argparse
import subprocess
from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_predict import cmd_predict


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        model_name='resnet-ae-v1',
        input_dir=None,
        output_dir=None,
        batch_size=32,
        save_latents=False,
        save_reconstructions=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_run_factory(returncode: int = 0):
    calls = []

    def _fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode)

    _fake_run.calls = calls
    return _fake_run


def _make_model_dir(results_dir: Path, model_name: str = 'resnet-ae-v1') -> Path:
    model_dir = results_dir / 'beast_models' / model_name
    model_dir.mkdir(parents=True)
    return model_dir


class TestCmdPredict:
    """Test the function cmd_predict."""

    def test_cmd_predict_beast_not_on_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr('cuttle_patterns.cli.cmd_predict.shutil.which', lambda name: None)
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_predict(args)
        assert exc_info.value.code == 1
        assert 'beast not found on PATH' in capsys.readouterr().out

    def test_cmd_predict_missing_model_dir(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_predict(args)
        assert exc_info.value.code == 1
        assert 'no model found at' in capsys.readouterr().out

    def test_cmd_predict_builds_expected_argv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        model_dir = _make_model_dir(results_dir)
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_predict.subprocess.run', fake_run)
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_predict(args)

        # Assert
        assert fake_run.calls[0] == [
            'beast', 'predict',
            '--model', str(model_dir),
            '--input', str(results_dir / 'beast_frames'),
            '--batch-size', '32',
        ]

    def test_cmd_predict_save_flags_and_output_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_model_dir(results_dir)
        output_dir = tmp_path / 'predictions'
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_predict.subprocess.run', fake_run)
        args = _make_args(
            results_dir=results_dir,
            output_dir=output_dir,
            save_latents=True,
            save_reconstructions=True,
        )

        # Act
        cmd_predict(args)

        # Assert
        argv = fake_run.calls[0]
        assert '--output' in argv
        assert argv[argv.index('--output') + 1] == str(output_dir)
        assert '--save_latents' in argv
        assert '--save_reconstructions' in argv

    def test_cmd_predict_no_save_flags_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_model_dir(results_dir)
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_predict.subprocess.run', fake_run)
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_predict(args)

        # Assert
        argv = fake_run.calls[0]
        assert '--save_latents' not in argv
        assert '--save_reconstructions' not in argv
        assert '--output' not in argv

    def test_cmd_predict_nonzero_returncode_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        _make_model_dir(results_dir)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_predict.subprocess.run', _fake_run_factory(returncode=2),
        )
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_predict(args)
        assert exc_info.value.code == 2
