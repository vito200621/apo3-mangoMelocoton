import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from src.training.group_cv import build_group_cv_splits


def test_group_cv_keeps_groups_separate_and_stratifies_each_fold():
    y = np.repeat(np.arange(3), 3)
    groups = np.array([f'{label}-{fold}' for label in range(3) for fold in range(3)])

    splits = build_group_cv_splits(y, groups)

    assert len(splits) == 3
    for train_indices, validation_indices in splits:
        assert set(y[train_indices]) == {0, 1, 2}
        assert set(y[validation_indices]) == {0, 1, 2}
        assert not set(groups[train_indices]) & set(groups[validation_indices])


def test_group_cv_requires_enough_independent_groups_per_class():
    y = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array(['a', 'a', 'b', 'c', 'd', 'e'])

    with pytest.raises(ValueError, match='distinct groups per class'):
        build_group_cv_splits(y, groups)


def test_grid_search_accepts_precomputed_group_folds():
    y = np.repeat(np.arange(3), 3)
    groups = np.array([f'{label}-{fold}' for label in range(3) for fold in range(3)])
    X = np.repeat(np.eye(3), 3, axis=0)
    splits = build_group_cv_splits(y, groups)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [1, 2]},
        cv=splits,
        scoring='f1_macro',
    )

    search.fit(X, y, groups=groups)

    assert search.best_score_ == 1.0
