import ctypes
import os
import numpy as np
from numpy.ctypeslib import ndpointer


class CppAuc:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings.
    """

    def __init__(self):
        try:
            self._handle = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "cpp_auc.so")
        except OSError:
            command_to_compile = f"cd {os.path.dirname(os.path.realpath(__file__))} && .{os.path.sep}compile.sh"
            raise OSError(f'Missing cpp_auc.so compiled file... '
                          f'You must first compile the C++ code to use this metric. '
                          f'Run the below terminal command to compile the code:\n'
                          f'{command_to_compile}')
        self._handle.cpp_auc_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t
                                             ]
        self._handle.cpp_auc_ext.restype = ctypes.c_double

    def roc_auc_score(self, y_true: np.array, y_score: np.array) -> float:
        """a method to calculate AUC via C++ lib.
        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool8 as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.
            sample_weight (np.array): 1D numpy array as sample weights, optional.
        Returns:
            float: AUC score
        """
        n = len(y_true)
        result = self._handle.cpp_auc_ext(y_score.astype(np.float32), y_true, n)
        return result
