"""Test for SymPy bug with TransferFunctionMatrix.rewrite(StateSpace).

See: https://github.com/sympy/sympy/issues/26827

Issue raised 2025-11-30:
 - https://github.com/sympy/sympy/issues/28678

When this test starts failing (i.e., no AttributeError is raised),
it means the SymPy bug has been fixed.
"""

import pytest
from sympy.abc import s
from sympy.physics.control.lti import (
    StateSpace,
    TransferFunction,
    TransferFunctionMatrix,
)


def test_sympy_transfer_function_matrix_to_state_space_bug():
    """Test that TransferFunctionMatrix.rewrite(StateSpace) raises AttributeError.

    This is a known bug in SymPy. When this test fails (no exception raised),
    the bug has been fixed and we can use TransferFunctionMatrix conversion.
    """
    tf1 = TransferFunction(1, s + 1, s)
    tf2 = TransferFunction(1, 2 * s + 1, s)
    G_tf = TransferFunctionMatrix([[tf1], [tf2]])

    # This should raise AttributeError due to SymPy bug
    with pytest.raises(
        AttributeError, match="'StateSpace' object has no attribute 'var'"
    ):
        G_tf.rewrite(StateSpace)

    # StateSpace constructor raises a different AttributeError
    with pytest.raises(
        AttributeError,
        match="'TransferFunctionMatrix' object has no attribute 'rows'",
    ):
        StateSpace(G_tf)
