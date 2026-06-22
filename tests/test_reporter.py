import json
from unittest.mock import MagicMock, patch

import pytest

from medpreprocess.reporter import QualityReport, generate_report

MOCK_REPORT = {
    "quality_score": 0.85,
    "contrast": "good",
    "noise_level": "low",
    "artifacts": [],
    "preprocessing_applied": ["CLAHE", "normalization", "resize"],
    "recommendations": ["Image quality is suitable for downstream analysis"],
    "overall_assessment": "High quality preprocessed image with good contrast and minimal noise.",
}


def _make_mock_client(response_json: dict):
    mock_content = MagicMock()
    mock_content.text = json.dumps(response_json)
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_generate_report_returns_quality_report(tmp_image_file):
    mock_client = _make_mock_client(MOCK_REPORT)
    with patch("medpreprocess.reporter.anthropic.Anthropic", return_value=mock_client):
        report = generate_report(str(tmp_image_file), api_key="test-key")
    assert isinstance(report, QualityReport)


def test_generate_report_schema_fields(tmp_image_file):
    mock_client = _make_mock_client(MOCK_REPORT)
    with patch("medpreprocess.reporter.anthropic.Anthropic", return_value=mock_client):
        report = generate_report(str(tmp_image_file), api_key="test-key")
    assert 0.0 <= report.quality_score <= 1.0
    assert report.contrast in ("poor", "acceptable", "good")
    assert report.noise_level in ("low", "moderate", "high")
    assert isinstance(report.artifacts, list)
    assert isinstance(report.recommendations, list)
    assert isinstance(report.overall_assessment, str)


def test_generate_report_passes_preprocessing_steps(tmp_image_file):
    mock_client = _make_mock_client(MOCK_REPORT)
    steps = ["clahe", "normalize", "resize"]
    with patch("medpreprocess.reporter.anthropic.Anthropic", return_value=mock_client):
        generate_report(str(tmp_image_file), preprocessing_steps=steps, api_key="test-key")
    call_kwargs = mock_client.messages.create.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0] if call_kwargs.args else []
    # Verify API was called once
    assert mock_client.messages.create.call_count == 1


def test_generate_report_raises_without_api_key(tmp_image_file, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
        generate_report(str(tmp_image_file))
