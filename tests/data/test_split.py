from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cxr_detect.data.split import (
    get_unique_patient_ids,
    split_patient_ids,
    filter_by_patient_ids,
    save_split,
)


def test_get_unique_patient_ids_returns_unique_patient_ids():
    df = pd.DataFrame({"patient_id": [1, 1, 2, 2, 3]})

    result = get_unique_patient_ids(df)
    assert isinstance(result, np.ndarray)
    assert set(result) == {1, 2, 3}


def test_get_unique_patient_ids_raises_if_missing_column():
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})

    with pytest.raises(KeyError):
        get_unique_patient_ids(df)


def test_split_patient_ids_sizes():
    ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    train, val, test = split_patient_ids(ids, train_ratio=0.6, val_ratio=0.2)

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2


def test_split_patient_ids_no_overlap():
    ids = np.arange(20)

    train, val, test = split_patient_ids(ids, seed=42)

    assert set(train).isdisjoint(set(val))
    assert set(val).isdisjoint(set(test))
    assert set(test).isdisjoint(set(train))


def test_split_patient_ids_is_deterministic():
    ids = np.arange(10)

    split1 = split_patient_ids(ids, seed=123)
    split2 = split_patient_ids(ids, seed=123)

    for a, b in zip(split1, split2):
        assert np.array_equal(a, b)


def test_filter_by_patient_ids():
    df = pd.DataFrame({"patient_id": [1, 2, 3, 4], "value": [10, 20, 30, 40]})

    filtered = filter_by_patient_ids(df, np.array([1, 3]))

    assert len(filtered) == 2
    assert set(filtered["patient_id"]) == {1, 3}


def test_save_split_creates_file(tmp_path):
    df = pd.DataFrame({"patient_id": [1, 2, 3], "value": [10, 20, 30]})

    patient_ids = np.array([1, 2])

    with patch("cxr_detect.data.split.save_processed_data") as mock_save:
        save_split(df, patient_ids, "train", tmp_path)

        assert mock_save.called


def test_save_split_raises_if_empty(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": [1, 2],
        }
    )

    with pytest.raises(ValueError):
        save_split(df, np.array([99]), "train", tmp_path)
