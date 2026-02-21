import numpy as np
import pandas as pd
import pytest

from cxr_detect.data.encode_labels import encode_labels


# Mock disease list
DISEASES = ["Atelectasis", "Cardiomegaly", "Effusion"]


def make_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Image Index": ["img1.png"],
            "Patient ID": [123],
            "Finding Labels": ["Atelectasis|Effusion"],
            "Patient Age": ["045Y"],
            "Patient Gender": ["M"],
            "View Position": ["PA"],
        }
    )


# ----------------------------------------
# Test 1: Required columns validation
# ----------------------------------------
def test_missing_required_columns():
    df = pd.DataFrame({"Image Index": ["img1.png"]})

    with pytest.raises(ValueError):
        encode_labels(df, DISEASES)


# ----------------------------------------
# Test 2: Column renaming works
# ----------------------------------------
def test_column_renaming():
    df = make_valid_dataframe()
    result = encode_labels(df, DISEASES)

    assert "image_id" in result.columns
    assert "patient_id" in result.columns
    assert "patient_age" in result.columns
    assert "patient_gender" in result.columns
    assert "view_position" in result.columns


# ----------------------------------------
# Test 3: Age conversion
# ----------------------------------------
def test_age_conversion():
    df = make_valid_dataframe()
    result = encode_labels(df, DISEASES)

    assert result["patient_age"].iloc[0] == 45
    assert isinstance(result["patient_age"].iloc[0], np.int64)


# ----------------------------------------
# Test 4: One-hot encoding correctness
# ----------------------------------------
def test_one_hot_encoding():
    df = make_valid_dataframe()
    result = encode_labels(df, DISEASES)

    assert result["Atelectasis"].iloc[0] == 1
    assert result["Cardiomegaly"].iloc[0] == 0
    assert result["Effusion"].iloc[0] == 1


# ----------------------------------------
# Test 5: Unknown disease should be ignored
# ----------------------------------------
def test_unknown_disease_ignored():
    df = make_valid_dataframe()
    df["Finding Labels"] = ["UnknownDisease"]

    result = encode_labels(df, DISEASES)

    assert result["Atelectasis"].iloc[0] == 0
    assert result["Cardiomegaly"].iloc[0] == 0
    assert result["Effusion"].iloc[0] == 0


# ----------------------------------------
# Test 6: Missing Finding Labels
# ----------------------------------------
def test_missing_finding_labels():
    df = make_valid_dataframe()
    df["Finding Labels"] = None

    result = encode_labels(df, DISEASES)

    assert result["Atelectasis"].iloc[0] == 0
    assert result["Cardiomegaly"].iloc[0] == 0
    assert result["Effusion"].iloc[0] == 0


# ----------------------------------------
# Test 7: Column dropping works safely
# ----------------------------------------
def test_columns_dropped():
    df = make_valid_dataframe()
    df["Follow-up #"] = 1

    result = encode_labels(df, DISEASES)

    assert "Follow-up #" not in result.columns
