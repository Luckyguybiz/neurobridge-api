"""Tests for ML pipeline module."""

from analysis.ml_pipeline import detect_anomalies, classify_states, compute_pca_embedding, extract_features


def test_extract_features(synthetic_data):
    result = extract_features(synthetic_data)
    assert isinstance(result, dict)
    assert "features" in result or "feature_names" in result


def test_anomalies(synthetic_data):
    result = detect_anomalies(synthetic_data)
    assert isinstance(result, dict)
    assert "anomalies" in result or "scores" in result


def test_classify_states(synthetic_data):
    result = classify_states(synthetic_data)
    assert isinstance(result, dict)


def test_pca_embedding(synthetic_data):
    result = compute_pca_embedding(synthetic_data)
    assert isinstance(result, dict)
    assert "embedding" in result or "components" in result or "explained_variance" in result
