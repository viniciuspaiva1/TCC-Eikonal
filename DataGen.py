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

    dados = [{'x': x, 'z': z, 'travel_time': travelTimes[x][z],'x0': sourceOrigin[0],'z0': sourceOrigin[1]} 
         for x in range(travelTimes.shape[0]) 
         for z in range(travelTimes.shape[1])]

    df = pd.DataFrame(dados)

    return df

def generateMultipleData(nPoints: tuple, nodeIntervals: tuple, velocityModel: np.ndarray,sourceOriginInterval: np.array) -> pd.DataFrame:
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
            frames.append(generateDataFrame(nPoints, nodeIntervals, velocityModel, (x,z,0)))
    
    return pd.concat(frames)