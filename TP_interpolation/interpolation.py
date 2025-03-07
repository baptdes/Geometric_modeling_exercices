
import math
import numpy as np
from typing import List, Tuple

#####################################################
############## Neville's algorithm ##################
#####################################################


def neville(XX: np.ndarray, YY:np.ndarray, x: float) -> float:
    """
    Perform polynomial interpolation using Neville's algorithm.

    Args:
        XX (np.ndarray): The x-coordinates of the data points.
        YY (np.ndarray): The y-coordinates of the data points.
        x (float): The value at which the interpolation is calculated.

    Returns:
        float: The interpolated value at x.
    """
    P = np.array(YY).copy()
    n = len(XX)
    for j in range(1, n):
        for i in range(n-j):
            P[i] = ((XX[i+j]-x)*P[i]+(x-XX[i])*P[i+1])/(XX[i+j]-XX[i])
    return P[0]
    

#####################################################
################ Time sampling ######################
#####################################################



def tchebycheff_parametrisation(nb_point: int) -> List[float]:
    """
    Compute the Tchebycheff abscissas for a given number of points (from -1 to 1).

    Args:
        nb_point (int): The number of points for which the Tchebycheff abscissas must be computed.

    Returns:
        List[float]: A list containing the Tchebycheff abscissas.
    """
    T = []
    for i in range(1,nb_point+1):
        T.append(math.cos((2*i-1)*math.pi/(2*nb_point)))
    return T



def regular_parametrisation(nb_point: int) -> List[float]:
    """
    Create regular subdivision with the first point at 0 and the last at 1.

    Args:
        nb_point (int): The number of points for which regular abscissas are calculated.

    Returns:
        List[float]: A list containing the regular abscissas.
    """
    return [i/(nb_point-1) for i in range(nb_point)]

def distance_parametrisation(XX: List[float], YY: List[float]) -> List[float]:
    """
    Create subdivision where spacing between points is proportional to their distance in R2,
    with the first point at 0 and the last at 1.

    Args:
        XX (List[float]): The X coordinates of the points.
        YY (List[float]): The Y coordinates of the points.

    Returns:
        List[float]: A list containing the abscissas proportional to the distances.
    """
    L = [0]
    for i in range(1,len(XX)):
        L.append(L[-1] + math.sqrt((XX[i]-XX[i-1])**2+(YY[i]-YY[i-1])**2))
    L = [l/L[-1] for l in L]
    return L


def parametrisation_racinedistance(XX: List[float], YY: List[float]) -> List[float]:
    """
    Create subdivision where spacing between points is proportional to the square root of their distance in R2,
    with the first point at 0 and the last at 1.

    Args:
        XX (List[float]): The X coordinates of the points.
        YY (List[float]): The Y coordinates of the points.

    Returns:
        List[float]: A list containing the abscissas proportional to the square roots of the distances.
    """
    L = [0]
    for i in range(1,len(XX)):
        L.append(L[-1] + math.sqrt(math.sqrt((XX[i]-XX[i-1])**2+(YY[i]-YY[i-1])**2)))
    L = [l/L[-1] for l in L]
    return L

#####################################################
############ Neville's Algorithm @@@@ ###############
#####################################################

def neville_param(XX, YY, TT, list_tt) -> Tuple[List[float], List[float]]:
    """
    Interpolate points using Neville's algorithm for given x and y coordinates.

    Args:
        XX (list[float]): The x-coordinates of the data points.
        YY (list[float]): The y-coordinates of the data points.
        TT (list[float]): The time points corresponding to the data points.
        list_tt (list[float]): The time points at which to evaluate the interpolation.

    Returns:
        tuple: Two lists containing the interpolated x and y coordinates.
    """
    interpolated_x = []
    interpolated_y = []
    for t in list_tt:
        x = neville(TT, XX, t)
        y = neville(TT, YY, t)
        interpolated_x.append(x)
        interpolated_y.append(y)
    return (interpolated_x, interpolated_y)


def surface_interpolation_neville(X, Y, Z, TT_x,TT_y, list_tt, nb_points_x) -> np.ndarray:
    """
    Interpolates a surface at given time points using Neville's interpolation method.

    Args:
        X (np.ndarray): X coordinates of 3D points (shape: (nb_points_x, len(TT_y))).
        Y (np.ndarray): Y coordinates of 3D points (shape: (nb_points_x, len(TT_y))).
        Z (np.ndarray): Z coordinates of 3D points (shape: (nb_points_x, len(TT_y))).
        TT_x (List[float]): Times corresponding to the points in X.
        TT_y (List[float]): Times corresponding to the points in Y.
        list_tt (List[float]): Times at which to evaluate the interpolated surface.
        nb_points_x (int): Number of points in the grid for interpolation.

    Returns:
        np.ndarray: Interpolated surface (shape: (len(list_tt), len(list_tt), 3)).
    """
    
    n = len(list_tt)
    interpolated_surface_1 = np.zeros((nb_points_x, n, 3))  # Intermediate surface
    interpolated_surface = np.zeros((n, n, 3))  # Final interpolated surface

    for i in range(nb_points_x):
        for j in range(n):
            x = neville(TT_y, X[i], list_tt[j])
            y = neville(TT_y, Y[i], list_tt[j])
            z = neville(TT_y, Z[i], list_tt[j])
            interpolated_surface_1[i][j] = [x, y, z]

    for i in range(n):
        for j in range(n):
            x = neville(TT_x, [interpolated_surface_1[k][i][0] for k in range(nb_points_x)], list_tt[j])
            y = neville(TT_x, [interpolated_surface_1[k][i][1] for k in range(nb_points_x)], list_tt[j])
            z = neville(TT_x, [interpolated_surface_1[k][i][2] for k in range(nb_points_x)], list_tt[j])
            interpolated_surface[i][j] = [x, y, z]

    return interpolated_surface