import numpy as np


def indices_to_artifact_dims(indices):
    """
    Converts indices returned by the algorithm to
    geometric dimension values in micrometers

    :param indices: indices of the PnC artifact
                s = [d, lxy],
                d --> index of diameter
                lxy --> index of distance
    :return:
        artifact_dimension: dimension of PnC artifact in micrometers
    """
    lxy = np.arange(700, 1100, 50)
    dia = np.arange(350, 650, 50)
    artifact_dimension = [dia[indices[0]],
                          lxy[indices[1]]]

    return artifact_dimension
