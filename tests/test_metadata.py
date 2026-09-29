"""Tests for cuttle_patterns.metadata."""

import pandas as pd
import pytest

from cuttle_patterns.metadata import attach_individual_column, load_identity_mapping


class TestLoadIdentityMapping:
    """Test the function load_identity_mapping."""

    def test_load_identity_mapping_packaged_file(self):
        # Act
        mapping = load_identity_mapping()

        # Assert
        assert list(mapping.columns) == ['session', 'date', 'day', 'tank', 'resident', 'intruder']
        assert len(mapping) == 72
        assert not mapping.duplicated(['day', 'tank']).any()
        assert set(mapping['resident']) == {f'R{idx}' for idx in range(1, 7)}
        assert set(mapping['intruder']) == {f'I{idx}' for idx in range(1, 7)}

    def test_load_identity_mapping_residents_fixed_to_tank(self):
        # Act
        mapping = load_identity_mapping()

        # Assert
        assert (mapping['resident'] == 'R' + mapping['tank'].astype(str)).all()

    def test_load_identity_mapping_spot_check(self):
        # Act
        mapping = load_identity_mapping().set_index(['day', 'tank'])

        # Assert
        assert mapping.loc[(3, 5), 'intruder'] == 'I1'
        assert mapping.loc[(12, 6), 'intruder'] == 'I5'


class TestAttachIndividualColumn:
    """Test the function attach_individual_column."""

    @pytest.fixture
    def mapping(self) -> pd.DataFrame:
        return pd.DataFrame({
            'session': ['s1', 's2'],
            'date': ['2025-06-02', '2025-06-03'],
            'day': [1, 2],
            'tank': [1, 1],
            'resident': ['R1', 'R1'],
            'intruder': ['I1', 'I2'],
        })

    def test_attach_individual_column_resident_and_intruder(self, mapping: pd.DataFrame):
        # Arrange
        meta = pd.DataFrame({
            'day': [2, 1, 2, 1],
            'tank': [1, 1, 1, 1],
            'role': ['Intruder', 'Resident', 'Resident', 'Intruder'],
            'frame_number': [3, 1, 2, 0],
        }, index=[10, 11, 12, 13])

        # Act
        out = attach_individual_column(meta, mapping)

        # Assert
        assert list(out['individual']) == ['I2', 'R1', 'R1', 'I1']
        assert list(out.index) == [10, 11, 12, 13]
        assert list(out['frame_number']) == [3, 1, 2, 0]
        assert 'individual' not in meta.columns

    def test_attach_individual_column_overwrites_existing(self, mapping: pd.DataFrame):
        # Arrange
        meta = pd.DataFrame({
            'day': [1], 'tank': [1], 'role': ['Intruder'], 'individual': ['stale'],
        })

        # Act
        out = attach_individual_column(meta, mapping)

        # Assert
        assert list(out['individual']) == ['I1']

    def test_attach_individual_column_uses_packaged_mapping(self):
        # Arrange
        meta = pd.DataFrame({'day': [1], 'tank': [2], 'role': ['Intruder']})

        # Act
        out = attach_individual_column(meta)

        # Assert
        assert list(out['individual']) == ['I2']

    def test_attach_individual_column_missing_session(self, mapping: pd.DataFrame):
        # Arrange
        meta = pd.DataFrame({'day': [5], 'tank': [3], 'role': ['Resident']})

        # Act & Assert
        with pytest.raises(ValueError, match=r'\(5, 3\)'):
            attach_individual_column(meta, mapping)

    def test_attach_individual_column_unknown_role(self, mapping: pd.DataFrame):
        # Arrange
        meta = pd.DataFrame({'day': [1], 'tank': [1], 'role': ['Bystander']})

        # Act & Assert
        with pytest.raises(ValueError, match='Bystander'):
            attach_individual_column(meta, mapping)
