"""Tests for cuttle_patterns.cli.cmd_extract_kmeans_video."""

from pathlib import Path

import pytest
import yaml

from cuttle_patterns.cli.cmd_extract_kmeans_video import cmd_extract_kmeans_video, register

STEMS = ['Day1_Tank1_Cuttle1_Resident_Crop', 'Day2_Tank1_Cuttle1_Resident_Crop',
         'Day3_Tank1_Cuttle1_Resident_Crop']


def _setup(tmp_path: Path, with_files: bool = True) -> Path:
    results = tmp_path / 'results'
    elig = results / 'extraction' / 'eligibility'
    clf = results / 'classifications' / 'clf'
    emb = results / 'embeddings' / 'dinov3_vits16'
    for d in (elig, clf, emb):
        d.mkdir(parents=True)
    (clf / 'classifier.yaml').write_text(yaml.safe_dump({'classes': ['A', 'B']}))
    for stem in STEMS:
        (elig / f'{stem}.parquet').touch()
        if with_files:
            (clf / f'{stem}.parquet').touch()
            (emb / f'{stem}.npy').touch()
    return results


def _args(parse_cli, results: Path, *extra: str):
    return parse_cli(register, [
        'extract-kmeans-video', '--results-dir', str(results), '--data-dir', 'unused',
        '--name', 'sel', '--classifier-name', 'clf', '--budget', '100', *extra,
    ])


class TestCmdExtractKmeansVideo:
    """Test the function cmd_extract_kmeans_video."""

    def test_missing_config_and_overrides_exits_with_error(self, parse_cli, no_config):
        # Arrange
        args = parse_cli(register, [
            'extract-kmeans-video', '--name', 'n', '--classifier-name', 'c', '--budget', '1',
        ])

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_kmeans_video(args)
        assert exc_info.value.code == 1

    def test_default_excludes_day_one_and_passes_settings(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        results = _setup(tmp_path)
        calls = []

        def fake(*args, **kwargs):
            calls.append((args, kwargs))
            return {'classes': ['A', 'B'], 'quotas': [60, 40]}

        monkeypatch.setattr('cuttle_patterns.cli.cmd_extract_kmeans_video.run_kmeans_video', fake)

        # Act
        cmd_extract_kmeans_video(_args(parse_cli, results, '--quota-power', '1'))

        # Assert
        (stems, _, _, _, out_dir, classes, budget), kwargs = calls[0]
        assert stems == STEMS[1:]
        assert out_dir == results / 'extraction' / 'kmeans_video' / 'sel'
        assert classes == ['A', 'B']
        assert budget == 100
        assert kwargs['quota_power'] == 1.0
        assert kwargs['classifier_name'] == 'clf'
        assert 'next: cuttle extract-kmeans-pooled --name sel' in capsys.readouterr().out

    def test_exclude_days_can_be_changed(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        # Arrange
        results = _setup(tmp_path)
        calls = []
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_kmeans_video.run_kmeans_video',
            lambda *a, **k: calls.append(a) or {'classes': [], 'quotas': []},
        )

        # Act
        cmd_extract_kmeans_video(_args(parse_cli, results, '--exclude-days', '2'))

        # Assert
        assert calls[0][0] == [STEMS[0], STEMS[2]]

    def test_missing_input_files_stop_unless_skip_missing(
        self, parse_cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
    ):
        # Arrange
        results = _setup(tmp_path, with_files=False)
        monkeypatch.setattr(
            'cuttle_patterns.cli.cmd_extract_kmeans_video.run_kmeans_video',
            lambda *a, **k: pytest.fail('should not run'),
        )

        # Act & Assert
        with pytest.raises(SystemExit) as exc_info:
            cmd_extract_kmeans_video(_args(parse_cli, results))
        assert exc_info.value.code == 1
        assert 'lack classifier or embedding files' in capsys.readouterr().out

    def test_skip_missing_with_nothing_left_prints_message(
        self, parse_cli, tmp_path: Path, capsys,
    ):
        # Arrange
        results = _setup(tmp_path, with_files=False)

        # Act
        cmd_extract_kmeans_video(_args(parse_cli, results, '--skip-missing'))

        # Assert
        assert 'No videos to process' in capsys.readouterr().out
