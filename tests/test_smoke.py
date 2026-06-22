"""End-to-end smoke test: verifies medpreprocess run works from a clean state."""
from pathlib import Path

from click.testing import CliRunner

from medpreprocess.cli import cli


def test_cli_run_end_to_end(tmp_image_file, tmp_path):
    runner = CliRunner()
    config = str(Path(__file__).parent.parent / "configs" / "xray_standard.yaml")
    result = runner.invoke(cli, [
        "run",
        "--config", config,
        "--input", str(tmp_image_file),
        "--output", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    output_files = list(tmp_path.iterdir())
    assert any("_preprocessed" in f.name for f in output_files)
    assert any("_meta.json" in f.name for f in output_files)


def test_cli_run_invalid_config_exits_with_error(tmp_image_file, tmp_path, tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("cfg")
    bad_config = config_dir / "bad.yaml"
    bad_config.write_text("steps:\n  - name: not_a_real_step\n")
    runner = CliRunner()
    result = runner.invoke(cli, [
        "run",
        "--config", str(bad_config),
        "--input", str(tmp_image_file),
        "--output", str(tmp_path),
    ])
    assert result.exit_code != 0
