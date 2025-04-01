# Import modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pykonal

#imShow para plotar as imagens em diagrama de cores
#Contours plot no plotly
#MatplotLib

# Instantiate EikonalSolver object using Cartesian coordinates.
solver = pykonal.EikonalSolver(coord_sys="cartesian")
# Set the coordinates of the lower bounds of the computational grid.
# For Cartesian coordinates this is x_min, y_min, z_min.
# In this example, the origin is the lower bound of the computation grid.
solver.velocity.min_coords = 0, 0, 0
# Set the interval between nodes of the computational grid.
# For Cartesian coordinates this is dx, dy, dz.
# In this example the nodes are separated by 1 km in in each direction.
solver.velocity.node_intervals = 1, 1, 1
# Set the number of nodes in the computational grid.
# For Cartesian coordinates this is nx, ny, nz.
# This is a 2D example, so we only want one node in the z direction.

solver.velocity.npts = 181, 361, 1
# solver.velocity.npts = 8, 8, 1

# Set the velocity model.
# In this case the velocity is equale to 1 km/s everywhere.

dg_over = 10 / 1000
xover = np.arange(0, 3600 / 1000 + dg_over, dg_over)
zover = np.arange(0, 1800 / 1000 + dg_over, dg_over)
v_line_z = 1.5 + zover
v_line_x = np.tile(0.3 * xover, (len(zover), 1))
v_depth = np.tile(v_line_z[:, np.newaxis], (1, len(xover))) + v_line_x

# #v_line_z_expanded = v_line_z[:, np.newaxis]

# # # Repetir ao longo da segunda dimensão para (181, 361)
# v_line_z_matrix = np.tile(v_line_z, (1, len(xover)))

# # # Expandir para (181, 361, 1)
# v_line_z_matrix = v_line_z_matrix.reshape(181,361,1)

# #print(v_line_z_matrix)  # Verificar (181, 361, 1)

# v_depth = v_depth.reshape(181,361,1)
v_depth = v_depth.reshape(181,361,1)


solver.velocity.values = v_depth

# Initialize the source. This is the trickiest part of the example.
# The source coincides with the node at index (0, 0, 0)
src_idx = 90, 180, 0
# Set the traveltime at the source node to 0.
solver.traveltime.values[src_idx] = 0
# Set the unknown flag for the source node to False.
# This is an FMM state variable indicating which values are completely
# unknown. Setting it to False indicates that the node has a tentative value
# assigned to it. In this case, the tentative value happens to be the true,
# final value.
solver.unknown[src_idx] = False
# Push the index of the source node onto the narrow-band heap.
solver.trial.push(*src_idx)

# Solve the system.
solver.solve()
# array_squeezed = np.squeeze(solver.traveltime.values)
df = pd.DataFrame(np.squeeze(solver.traveltime.values))

df.to_csv("dados.csv", index=False)

# for a in array_squeezed:
#     # for b in a:
#     #     print(f'{b} ')
#     print(a)


# And finally, get the traveltime values.
#print(solver.traveltime.values)

plt.imshow(solver.traveltime.values, cmap='terrain', interpolation='none')
plt.colorbar()  # Exibe a barra de cores para referência
plt.title("Heatmap da Matriz")
plt.show()