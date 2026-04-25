"""Shared pytest fixtures for tests"""

import pytest
import numpy as np


@pytest.fixture
def tf_test_case_1():
    """Second order system - observable canonical form matching Octave.

    Input: num=[0.2, 0.1], den=[1, -1.4, 0.49]
    """
    return {
        "num": np.array([0.2, 0.1]),
        "den": np.array([1, -1.4, 0.49]),
        "A": np.array([[0.0, -0.49], [1.0, 1.4]]),
        "B": np.array([[0.1], [0.2]]),
        "C": np.array([[0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_2():
    """Third order system - observable canonical form matching Octave.

    Input: num=[0.05, 0.1, 0.05], den=[1, -1.5, 0.7, -0.1]
    """
    return {
        "num": np.array([0.05, 0.1, 0.05]),
        "den": np.array([1, -1.5, 0.7, -0.1]),
        "A": np.array([[0.0, 0.0, 0.1], [1.0, 0.0, -0.7], [0.0, 1.0, 1.5]]),
        "B": np.array([[0.05], [0.1], [0.05]]),
        "C": np.array([[0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_3():
    """Fifth order system - observable canonical form matching Octave.

    Input: num=[0.01, 0.03, 0.03, 0.02, 0.01],
           den=[1, -2.3, 2.6, -1.8, 0.72, -0.12]
    """
    return {
        "num": np.array([0.01, 0.03, 0.03, 0.02, 0.01]),
        "den": np.array([1, -2.3, 2.6, -1.8, 0.72, -0.12]),
        "A": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.12],
                [1.0, 0.0, 0.0, 0.0, -0.72],
                [0.0, 1.0, 0.0, 0.0, 1.8],
                [0.0, 0.0, 1.0, 0.0, -2.6],
                [0.0, 0.0, 0.0, 1.0, 2.3],
            ]
        ),
        "B": np.array([[0.01], [0.02], [0.03], [0.03], [0.01]]),
        "C": np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_4():
    """First order system - observable canonical form matching Octave.

    Input: num=[1], den=[1, -1]
    """
    return {
        "num": np.array([1]),
        "den": np.array([1, -1]),
        "A": np.array([[1.0]]),
        "B": np.array([[1.0]]),
        "C": np.array([[1.0]]),
        "D": np.array([[0.0]]),
    }


@pytest.fixture
def tf_test_case_5():
    """Sixth order system - observable canonical form matching Octave.

    Input: num=[0, 0, 0, 0, 0, 0, 0.1],
           den=[1.0, -4.514214, 8.884062, -9.749747, 6.204163, -2.124264, 0.3]
    """
    return {
        "num": np.array([0, 0, 0, 0, 0, 0, 0.1000]),
        "den": np.array(
            [
                1.000000,
                -4.514214,
                8.884062,
                -9.749747,
                6.204163,
                -2.124264,
                0.300000,
            ]
        ),
        "A": np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.3],
                [1.0, 0.0, 0.0, 0.0, 0.0, 2.124264],
                [0.0, 1.0, 0.0, 0.0, 0.0, -6.204163],
                [0.0, 0.0, 1.0, 0.0, 0.0, 9.749747],
                [0.0, 0.0, 0.0, 1.0, 0.0, -8.884062],
                [0.0, 0.0, 0.0, 0.0, 1.0, 4.514214],
            ]
        ),
        "B": np.array([[0.1], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        "C": np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        "D": np.array([[0.0]]),
    }
