# Script Blender: Creeaza un grid
# Intent: create_grid
# Parametri: size=4, subdivisions=10

import bpy

bpy.ops.mesh.primitive_grid_add(
    x_subdivisions=10,
    y_subdivisions=10,
    size=4,
    location=(0, 0, 0)
)

obj = bpy.context.active_object
obj.name = "Grid_Generated"

print("Grid creat!")
