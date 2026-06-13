"""Tests for survey analysis tools."""

import pandas as pd
import pytest

from app.tools.anomaly_flag import flag_anomalies
from app.tools.segment_stats import get_segment_stats
from app.tools.trend_compare import compare_trends


@pytest.fixture
def sample_df():
    """Create a sample survey DataFrame."""
    return pd.DataFrame({
        "Department": ["Sales", "Marketing", "Sales", "Marketing", "Sales", "HR", "HR", "Sales"],
        "Quarter": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q1", "Q2", "Q3"],
        "Satisfaction": [4.0, 5.0, 3.0, 4.0, 4.5, 3.5, 4.0, 5.0],
        "NPS": [8, 9, 7, 8, 9, 6, 7, 9]
    })


@pytest.fixture
def sample_schema():
    """Create a sample schema."""
    return {
        "Department": {"type": "categorical", "n_unique": 3},
        "Quarter": {"type": "categorical", "n_unique": 3},
        "Satisfaction": {"type": "numeric_score", "n_unique": 6},
        "NPS": {"type": "numeric_scale", "n_unique": 4}
    }


class TestSegmentStats:
    """Tests for segment statistics tool."""
    
    def test_basic_segmentation(self, sample_df, sample_schema, monkeypatch):
        """Compute stats by segment."""
        monkeypatch.setattr("app.tools.segment_stats.MIN_SEGMENT_SIZE", 2)

        result = get_segment_stats(
            sample_df, sample_schema,
            segment_column="Department",
            metric_column="Satisfaction"
        )
        
        assert result["success"] is True
        assert result["segment_column"] == "Department"
        assert result["metric_column"] == "Satisfaction"
        assert len(result["segments"]) > 0
        assert "best_segment" in result
        assert "worst_segment" in result
    
    def test_missing_column_error(self, sample_df, sample_schema):
        """Handle missing columns gracefully."""
        result = get_segment_stats(
            sample_df, sample_schema,
            segment_column="NonExistent",
            metric_column="Satisfaction"
        )
        
        assert result["success"] is False
        assert "error" in result
    
    def test_non_numeric_metric_error(self, sample_df, sample_schema):
        """Reject non-numeric metrics."""
        result = get_segment_stats(
            sample_df, sample_schema,
            segment_column="Department",
            metric_column="Quarter"
        )
        
        assert result["success"] is False


class TestTrendCompare:
    """Tests for trend comparison tool."""
    
    def test_basic_comparison(self, sample_df, sample_schema):
        """Compare trends across quarters."""
        result = compare_trends(
            sample_df, sample_schema,
            dimension_column="Quarter",
            metric_columns=["Satisfaction", "NPS"]
        )
        
        assert result["success"] is True
        assert result["dimension_column"] == "Quarter"
        assert len(result["periods"]) > 0
        assert len(result["data"]) > 0
    
    def test_single_period_error(self, sample_df, sample_schema):
        """Handle single period data."""
        df_single = sample_df[sample_df["Quarter"] == "Q1"]
        
        result = compare_trends(
            df_single, sample_schema,
            dimension_column="Quarter",
            metric_columns=["Satisfaction"]
        )
        
        assert result["success"] is False


class TestAnomalyFlag:
    """Tests for anomaly detection tool."""
    
    def test_anomaly_detection(self, sample_df, sample_schema):
        """Detect outliers in data."""
        result = flag_anomalies(
            sample_df, sample_schema,
            columns=["Satisfaction", "NPS"]
        )
        
        assert result["success"] is True
        assert "anomalies_by_column" in result
    
    def test_invalid_column(self, sample_df, sample_schema):
        """Handle invalid columns."""
        result = flag_anomalies(
            sample_df, sample_schema,
            columns=["NonExistent"]
        )
        
        assert result["success"] is False


class TestToolIntegration:
    """Integration tests for tools."""
    
    def test_end_to_end_workflow(self, sample_df, sample_schema, monkeypatch):
        """Test complete analysis workflow."""
        monkeypatch.setattr("app.tools.segment_stats.MIN_SEGMENT_SIZE", 2)

        # Step 1: Segment analysis
        segment_result = get_segment_stats(
            sample_df, sample_schema,
            segment_column="Department",
            metric_column="Satisfaction"
        )
        assert segment_result["success"] is True
        
        # Step 2: Trend analysis
        trend_result = compare_trends(
            sample_df, sample_schema,
            dimension_column="Quarter",
            metric_columns=["Satisfaction"]
        )
        assert trend_result["success"] is True
        
        # Step 3: Anomaly detection
        anomaly_result = flag_anomalies(
            sample_df, sample_schema,
            columns=["Satisfaction", "NPS"]
        )
        assert anomaly_result["success"] is True
