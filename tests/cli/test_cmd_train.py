"""Tests for cuttle_patterns.cli.cmd_train."""

import argparse
import subprocess
from pathlib import Path

import pytest

from cuttle_patterns.cli.cmd_train import cmd_train


def _raise_file_not_found() -> None:
    raise FileNotFoundError('no config file found at ~/.cuttle-patterns/config.yaml')


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        results_dir=None,
        config=Path('/configs/beast_resnet_ae.yaml'),
        model_name='resnet-ae-v1',
        input_dir=None,
        gpus=None,
        nodes=None,
        overrides=None,
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


class TestCmdTrain:
    """Test the function cmd_train."""

    def test_cmd_train_missing_config_and_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.load_config', _raise_file_not_found)
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 1
        assert 'Error' in capsys.readouterr().out

    def test_cmd_train_beast_not_on_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.shutil.which', lambda name: None)
        args = _make_args()

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 1
        assert 'beast not found on PATH' in capsys.readouterr().out

    def test_cmd_train_builds_expected_argv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.subprocess.run', fake_run)
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_train(args)

        # Assert
        assert len(fake_run.calls) == 1
        argv = fake_run.calls[0]
        assert argv == [
            'beast', 'train',
            '--config', str(args.config),
            '--data', str(results_dir / 'beast_frames'),
            '--output', str(results_dir / 'beast_models' / 'resnet-ae-v1'),
        ]

    def test_cmd_train_gpus_nodes_overrides_passthrough(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.subprocess.run', fake_run)
        args = _make_args(
            results_dir=results_dir, gpus=4, nodes=2, overrides=['training.num_epochs=10'],
        )

        # Act
        cmd_train(args)

        # Assert
        argv = fake_run.calls[0]
        assert argv[-6:] == [
            '--gpus', '4', '--nodes', '2', '--overrides', 'training.num_epochs=10',
        ]

    def test_cmd_train_custom_input_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        input_dir = tmp_path / 'custom_frames'
        fake_run = _fake_run_factory()
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.subprocess.run', fake_run)
        args = _make_args(results_dir=results_dir, input_dir=input_dir)

        # Act
        cmd_train(args)

        # Assert
        argv = fake_run.calls[0]
        assert '--data' in argv
        assert argv[argv.index('--data') + 1] == str(input_dir)

    def test_cmd_train_nonzero_returncode_propagates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_train.subprocess.run', _fake_run_factory(returncode=3),
        )
        args = _make_args(results_dir=results_dir)

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 3

    def test_cmd_train_warns_on_nonempty_existing_model_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ):
        # Arrange
        results_dir = tmp_path / 'results'
        model_dir = results_dir / 'beast_models' / 'resnet-ae-v1'
        model_dir.mkdir(parents=True)
        (model_dir / 'training.log').write_text('existing')
        monkeypatch.setattr('cuttle_patterns.cli.cmd_train.subprocess.run', _fake_run_factory())
        args = _make_args(results_dir=results_dir)

        # Act
        cmd_train(args)

        # Assert
        assert 'already exists and is non-empty' in capsys.readouterr().out
