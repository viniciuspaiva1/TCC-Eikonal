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
    df = df.loc[~((df['x'] == df['source_x0']) & (df['z'] == df['source_z0']))] # remove os pontos na fonte

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
    print(xInterval)
    print(zInterval)

    frames = []
    
    for x in xInterval:
        for z in zInterval:
            frames.append(generateDataFrame(nPoints, nodeDistances, velocityModel, (x,z,0)))
    
    return pd.concat(frames)

def generateSpacedSourcesData(nPoints: tuple, nodeIntervals: tuple, velocityModel: np.ndarray, spacings: list) -> dict:
    """
    Gera 3 conjuntos de dados por grid, com fontes espaçadas uniformemente segundo os valores em `spacings`.

    Retorna um dicionário onde as chaves são os espaçamentos usados e os valores são os DataFrames.
    """
    max_x, max_z = nPoints[0], nPoints[1]
    data_by_spacing = {}

    for spacing in spacings:
        frames = []
        x_coords = np.arange(0, max_x, spacing)
        z_coords = np.arange(0, max_z, spacing)

        for x in x_coords:
            for z in z_coords:
                sourceOrigin = (x, z, 0)
                df = generateDataFrame(nPoints, nodeIntervals, velocityModel, sourceOrigin)
                frames.append(df)

        data_by_spacing[spacing] = pd.concat(frames, ignore_index=True)

    return data_by_spacing