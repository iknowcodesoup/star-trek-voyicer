"""Tests for stage_preprocess's fingerprint-based skip/regenerate decision.

cli.py imports chatterbox-tts, whisper, and piper_phonemize at module load
time (unlike app.py, see test_app.py's docstring) -- but those are ordinary
dependencies of this project's own uv environment, so importing cli.py
directly here needs no extra setup, just a normal `uv run pytest tests/`.
"""

from pathlib import Path

import pytest

from jeanlucrecord import cli


def write_metadata(directory: Path, rows: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "metadata.csv"
    metadata_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return metadata_path


def build_character(app_dir: Path, character: str, resampled_rows: list[str]) -> Path:
    resampled_dir = app_dir / "work" / character / "resampled"
    write_metadata(resampled_dir, resampled_rows)
    return resampled_dir


def write_config(training_dir: Path) -> None:
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "config.json").write_text("{}", encoding="utf-8")


# -- _metadata_fingerprint -----------------------------------------------


def test_metadata_fingerprint_reports_clip_count_and_a_content_hash(tmp_path):
    write_metadata(tmp_path, ["id1|line one", "id2|line two"])

    fingerprint = cli._metadata_fingerprint(tmp_path)

    assert fingerprint["clip_count"] == 2
    assert isinstance(fingerprint["metadata_hash"], str)
    assert fingerprint["metadata_hash"]


def test_metadata_fingerprint_changes_when_the_content_changes(tmp_path):
    write_metadata(tmp_path, ["id1|line one"])
    before = cli._metadata_fingerprint(tmp_path)

    write_metadata(tmp_path, ["id1|line one", "id2|line two"])
    after = cli._metadata_fingerprint(tmp_path)

    assert before != after


def test_metadata_fingerprint_raises_a_readable_error_when_metadata_csv_is_missing(
    tmp_path,
):
    with pytest.raises(SystemExit, match="resample"):
        cli._metadata_fingerprint(tmp_path)


def test_metadata_fingerprint_raises_a_readable_error_for_non_utf8_metadata_csv(
    tmp_path,
):
    """A raw UnicodeDecodeError must turn into the same clean SystemExit as
    the missing-file case, not an unhandled traceback."""
    (tmp_path / "metadata.csv").write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(SystemExit, match="resample"):
        cli._metadata_fingerprint(tmp_path)


# -- _load_sidecar / _write_sidecar --------------------------------------


def test_load_sidecar_returns_none_when_no_sidecar_exists(tmp_path):
    assert cli._load_sidecar(tmp_path) is None


def test_load_sidecar_treats_corrupt_json_as_absent(tmp_path):
    (tmp_path / cli.DATASET_FINGERPRINT_NAME).write_text(
        "{not valid json", encoding="utf-8"
    )

    assert cli._load_sidecar(tmp_path) is None


def test_load_sidecar_treats_invalid_utf8_bytes_as_absent(tmp_path):
    """A truncated write (crash, full disk, kill mid-write) can leave invalid
    UTF-8 bytes, not just syntactically-broken-but-valid-text JSON -- that
    must read back as "no sidecar" too, not raise UnicodeDecodeError."""
    (tmp_path / cli.DATASET_FINGERPRINT_NAME).write_bytes(b"\xff\xfe\x00\x01")

    assert cli._load_sidecar(tmp_path) is None


def test_write_sidecar_then_load_sidecar_round_trips(tmp_path):
    fingerprint = {"clip_count": 3, "metadata_hash": "abc123"}

    cli._write_sidecar(tmp_path, fingerprint)

    assert cli._load_sidecar(tmp_path) == fingerprint


def test_write_sidecar_leaves_no_temp_file_behind(tmp_path):
    """_write_sidecar writes to a .tmp file and replaces it into place --
    confirm the replace actually happens and doesn't leave the temp file
    sitting next to the real sidecar."""
    cli._write_sidecar(tmp_path, {"clip_count": 1, "metadata_hash": "x"})

    assert (tmp_path / cli.DATASET_FINGERPRINT_NAME).exists()
    assert not (tmp_path / (cli.DATASET_FINGERPRINT_NAME + ".tmp")).exists()


# -- stage_preprocess: skip/regenerate decision --------------------------


def test_stage_preprocess_runs_docker_and_writes_a_sidecar_on_first_run(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    resampled_dir = build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"
    calls = []

    def fake_run_docker(*args, **kwargs):
        calls.append(args)
        write_config(training_dir)

    monkeypatch.setattr(cli, "run_docker", fake_run_docker)

    cli.stage_preprocess("janeway")

    assert calls, "run_docker should have been called"
    assert cli._load_sidecar(training_dir) == cli._metadata_fingerprint(resampled_dir)
    assert "Regenerating" in capsys.readouterr().out


def test_stage_preprocess_skips_when_the_fingerprint_already_matches(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    resampled_dir = build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"
    write_config(training_dir)
    cli._write_sidecar(training_dir, cli._metadata_fingerprint(resampled_dir))

    def fail_run_docker(*args, **kwargs):
        raise AssertionError(
            "run_docker must not be called when the fingerprint matches"
        )

    monkeypatch.setattr(cli, "run_docker", fail_run_docker)

    cli.stage_preprocess("janeway")  # must not raise

    assert "skipping" in capsys.readouterr().out


def test_stage_preprocess_regenerates_when_new_clips_landed_in_resampled(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    resampled_dir = build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"
    write_config(training_dir)
    cli._write_sidecar(training_dir, cli._metadata_fingerprint(resampled_dir))

    # new clips reached resampled/ since the last preprocess run
    write_metadata(resampled_dir, ["id1|line one", "id2|line two"])
    calls = []
    monkeypatch.setattr(
        cli,
        "run_docker",
        lambda *args, **kwargs: (calls.append(args), write_config(training_dir)),
    )

    cli.stage_preprocess("janeway")

    assert calls
    assert cli._load_sidecar(training_dir) == cli._metadata_fingerprint(resampled_dir)


def test_stage_preprocess_regenerates_when_config_json_is_missing(
    tmp_path, monkeypatch
):
    """A partially-cleared training/ (sidecar present, config.json gone) must
    always regenerate, not silently stay stale."""
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    resampled_dir = build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"
    training_dir.mkdir(parents=True)
    cli._write_sidecar(training_dir, cli._metadata_fingerprint(resampled_dir))
    # config.json deliberately not written here

    calls = []
    monkeypatch.setattr(
        cli,
        "run_docker",
        lambda *args, **kwargs: (calls.append(args), write_config(training_dir)),
    )

    cli.stage_preprocess("janeway")

    assert calls


def test_stage_preprocess_ignores_dataset_changes_that_have_not_reached_resampled(
    tmp_path, monkeypatch
):
    """New clips landed in dataset/ but --stage resample has not rerun yet, so
    resampled/metadata.csv is unchanged -- nothing new actually reached
    preprocess's real input, so this must stay a no-op."""
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    resampled_dir = build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"
    write_config(training_dir)
    cli._write_sidecar(training_dir, cli._metadata_fingerprint(resampled_dir))

    # simulate new clips committed to dataset/ without an intervening
    # --stage resample
    write_metadata(
        tmp_path / "work" / "janeway" / "dataset",
        ["id1|line one", "id2|a brand new clip"],
    )

    def fail_run_docker(*args, **kwargs):
        raise AssertionError("run_docker must not be called: resampled/ did not change")

    monkeypatch.setattr(cli, "run_docker", fail_run_docker)

    cli.stage_preprocess("janeway")  # must not raise


def test_stage_preprocess_does_not_write_a_sidecar_when_run_docker_fails(
    tmp_path, monkeypatch
):
    """The sidecar is the record that preprocessing succeeded -- an
    interrupted or failed run must be retried next time, not mistaken for
    up to date."""
    monkeypatch.setattr(cli, "APP_DIR", tmp_path)
    build_character(tmp_path, "janeway", ["id1|line one"])
    training_dir = tmp_path / "work" / "janeway" / "training"

    def broken_run_docker(*args, **kwargs):
        raise RuntimeError("docker run failed")

    monkeypatch.setattr(cli, "run_docker", broken_run_docker)

    with pytest.raises(RuntimeError):
        cli.stage_preprocess("janeway")

    assert cli._load_sidecar(training_dir) is None
