import numpy as np
import pandas as pd
import pykonal

def generateDataFrame(nPoints: tuple, nodeIntervals: tuple, velocityModel: np.ndarray, sourceOrigin: tuple) -> pd.DataFrame:
    solver = pykonal.EikonalSolver(coord_sys="cartesian")

    solver.velocity.min_coords = 0, 0, 0 

    solver.velocity.node_intervals = [*nodeIntervals]
    solver.velocity.npts = [*nPoints]
    solver.velocity.values = velocityModel

    solver.traveltime.values[sourceOrigin] = 0
    solver.unknown[sourceOrigin] = False
    solver.trial.push(*sourceOrigin)

    solver.solve()

    travelTimes = np.squeeze(solver.traveltime.values) #Para remover a Terceira dimensão

    dados = [{'x': x, 'z': z, 'travel_time': travelTimes[x][z],'source_x0': sourceOrigin[0],'source_z0': sourceOrigin[1]} 
         for x in range(travelTimes.shape[0]) 
         for z in range(travelTimes.shape[1])]

    df = pd.DataFrame(dados)

    return df

def generateMultipleData(nPoints: tuple, nodeDistances: tuple, velocityModel: np.ndarray,sourceOriginInterval: np.array) -> pd.DataFrame:
    """
    - sourceOriginInterval: np.array -> Must be the interval of the source in x and z axis, 
    e.g: [[0,0], [2,10]]
    in this case, all the pairs (0,0), (0,1), (0,2)... going trough the 33 pairs 

        - x= [0,1,2], z = [0,1,2,3,4,5,6,7,8,9,10]
    """

    xInterval = np.arange(sourceOriginInterval[0][0], sourceOriginInterval[1][0] + 1)
    zInterval = np.arange(sourceOriginInterval[0][1], sourceOriginInterval[1][1] + 1)

    frames = []
    
    for x in xInterval:
        for z in zInterval:
            frames.append(generateDataFrame(nPoints, nodeDistances, velocityModel, (x,z,0)))
    
    return pd.concat(frames)

def check_derivatives(data: pd.DataFrame):
    """
    Check the derivatives of the traveltime field.
    """
    new_data = data.copy()
    
    # Avoid division by zero by replacing zero denominators with NaN temporarily
    dx = new_data['x'] - new_data['source_x0']
    dz = new_data['z'] - new_data['source_z0']
    
    new_data['dt_dx'] = new_data['travel_time'] / dx
    new_data['dt_dz'] = new_data['travel_time'] / dz

    # Replace inf or NaN with 0 after computation
    new_data['dt_dx'].replace([np.inf, -np.inf], 0, inplace=True)
    new_data['dt_dz'].replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate the velocity
    new_data['velocity_hat'] = 1 / np.sqrt(new_data['dt_dx']**2 + new_data['dt_dz']**2)
    new_data['velocity'] = 1.5 + 0.3 * new_data['z'] + new_data['x']

    return new_data