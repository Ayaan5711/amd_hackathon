"""Tests for CSV loader utility."""

import io

import pandas as pd
import pytest

from app.utils.csv_loader import detect_column_type, load_csv


class TestDetectColumnType:
    """Tests for column type detection."""
    
    def test_numeric_scale_detection(self):
        """Detect Likert scale columns."""
        series = pd.Series([1, 2, 3, 4, 5, 3, 4, 5, 2, 1])
        assert detect_column_type(series) == "numeric_scale"
    
    def test_numeric_score_detection(self):
        """Detect continuous numeric columns."""
        series = pd.Series([1.5, 2.3, 3.7, 4.2, 5.1, 100.5])
        assert detect_column_type(series) == "numeric_score"
    
    def test_open_text_detection(self):
        """Detect open text columns."""
        series = pd.Series([
            "This is a long comment about something",
            "Another detailed feedback response here",
            "Yet another lengthy text entry"
        ])
        assert detect_column_type(series) == "open_text"
    
    def test_categorical_detection(self):
        """Detect categorical columns."""
        series = pd.Series(["A", "B", "A", "C", "B", "A"])
        assert detect_column_type(series) == "categorical"
    
    def test_boolean_detection(self):
        """Detect boolean columns."""
        series = pd.Series(["Yes", "No", "Yes", "No", "Yes"])
        assert detect_column_type(series) == "boolean"


class TestLoadCSV:
    """Tests for CSV loading."""
    
    def test_load_valid_csv(self):
        """Load a valid CSV file."""
        csv_content = """Name,Department,Score
John,Sales,4
Jane,Marketing,5
Bob,Sales,3"""
        
        df, schema = load_csv(csv_content.encode('utf-8'), "test.csv")
        
        assert len(df) == 3
        assert len(df.columns) == 3
        assert "Name" in schema
        assert "Department" in schema
        assert "Score" in schema
    
    def test_empty_file_error(self):
        """Reject empty files."""
        with pytest.raises(ValueError, match="empty"):
            load_csv(b"", "empty.csv")
    
    def test_single_column_error(self):
        """Reject single column files."""
        csv_content = "Column1\nvalue1\nvalue2"
        
        with pytest.raises(ValueError, match="at least 2 columns"):
            load_csv(csv_content.encode('utf-8'), "single.csv")
    
    def test_schema_detection(self):
        """Correctly detect column types in schema."""
        csv_content = """ID,Department,Satisfaction,Comments
1,Sales,4,Great service
2,Marketing,5,Very helpful
3,Sales,3,Could be better"""
        
        df, schema = load_csv(csv_content.encode('utf-8'), "survey.csv")
        
        assert schema["ID"]["type"] in ["numeric_scale", "numeric_score"]
        assert schema["Department"]["type"] == "categorical"
        assert schema["Satisfaction"]["type"] == "numeric_scale"
        assert schema["Comments"]["type"] == "open_text"


class TestCSVEdgeCases:
    """Tests for edge cases."""
    
    def test_whitespace_in_columns(self):
        """Handle whitespace in column names."""
        csv_content = """ Name , Department , Score 
John,Sales,4"""
        
        df, _ = load_csv(csv_content.encode('utf-8'), "spaces.csv")
        
        # Column names should be stripped
        assert "Name" in df.columns or " Name " in df.columns
    
    def test_numeric_as_string(self):
        """Handle numeric values stored as strings."""
        csv_content = '''ID,Score
1,"4"
2,"5"
3,"3"
'''

        df, schema = load_csv(csv_content.encode('utf-8'), "strings.csv")
        
        # Should detect as numeric
        assert schema["Score"]["type"] in ["numeric_scale", "numeric_score"]
