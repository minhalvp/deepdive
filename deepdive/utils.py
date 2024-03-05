import importlib
import warnings


def import_cupy_else_numpy():
    """
      Tries to import the cupy module, if not installed, imports numpy.

      Returns
      -------
      module
          The imported module, either cupy or numpy.

      Raises
      ------
      ImportError
          If neither cupy nor numpy can be imported.
      """
    try:
        return importlib.import_module("cupy")
    except ImportError:
        warnings.warn(
            "CuPy not installed, falling back to NumPy", stacklevel=2)
        return importlib.import_module("numpy")


def check_cupy() -> bool:
    try:
        importlib.import_module('cupy')
        return True
    except ImportError:
        return False
