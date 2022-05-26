import sys
import numpy as np

# aux functions for the backend :9


def test_var_is_number(x):
    """
    Test if variable represents a real number
    """
    if type(x) == int or type(x) == float:
        return True
    else:
        return False


def param_name(param):
    """
    This function builds the label for a given parameter.
    """
    ptype, index = param.ptype, param.index
    if ptype == "c" and len(index) == 2:
        return "c" + str(index[0]) + str(index[1])
    elif ptype == "a" or ptype == "b":
        return ptype + str(index)
    elif ptype == "f" and index == 1:
        return "F"
    elif ptype == "f" and index == 2:
        return "w"
    elif ptype == "f" and index == 3:
        return "p"
    elif ptype == "x0" or ptype == "v0":
        return ptype
    else:
        sys.exit("tsaopy backend error : Error naming parameters.")


def param_cindex(param):
    """
    This function returns the location in the params array used by the backend for a given param.
    """
    if param.ptype == "x0":
        return 0, 0
    elif param.ptype == "v0":
        return 0, 1
    elif param.ptype == "a":
        return 1, param.index - 1
    elif param.ptype == "b":
        return 2, param.index - 1
    elif param.ptype == "c":
        q = param.index
        return 3, (q[0] - 1, q[1] - 1)
    elif param.ptype == "f":
        return 4, param.index - 1


def test_params_are_ok(params):
    """
    Test if a given list of tsaopy parameters meets the bare minimum to build a tsaopy model.
    """
    params_list = []
    for param in params:
        params_list.append(param_name(param))
    params_set = set(params_list)

    if len(params_list) != len(params_set):
        sys.exit("tsaopy model error: you have defined repeated parameters.")

    if ("x0" not in params_set) or ("v0" not in params_set):
        sys.exit("tsaopy model error: you haven't defined proper initial conditions in the"
                 "parameters.")

    if len(params_set) < 3:
        sys.exit("tsaopy model error: you haven't defined any ODE coefficients in the parameters.")


def ptype_array_shape(params, n_ptype):
    """
    This function gets the shape of the params array for the n ptype.
    """
    indexes = []
    for elem in params:
        if elem.ptype == n_ptype:
            indexes.append(elem.index)
    if n_ptype == "c" and not len(indexes) == 0:
        arraux = np.array(indexes)
        return max(arraux[:, 0]), max(arraux[:, 1])
    elif n_ptype == "c" and len(indexes) == 0:
        return (0, 0)
    elif (n_ptype == "a" or n_ptype == "b" or n_ptype == "f") and not len(
        indexes
    ) == 0:
        return (max(indexes),)
    elif (n_ptype == "a" or n_ptype == "b" or n_ptype == "f") and len(
        indexes
    ) == 0:
        return (0,)
    else:
        sys.exit("tsaopy backend error: n ptype is not a, b, c, or f when building the"
                 "parameters array.")


def params_array_shape(params):
    """
    This function builds an array that indicates the shape of all params arrays.
    """
    all_params_array_shape = [(1,), (1,)]

    for _ in ["a", "b", "c"]:
        all_params_array_shape.append(ptype_array_shape(params, _))

    # f array always has length 3

    all_params_array_shape.append((3,))

    return all_params_array_shape


def fitparams_info(fparams):
    """
    This function builds the list of labels for all fitting parameters, and indicates each
    parameter's position in the parameters array used in the simulations.
    """
    indexes, labels = [], []
    for fparam in fparams:
        indexes.append(param_cindex(fparam))
        labels.append(param_name(fparam))
    return indexes, labels
