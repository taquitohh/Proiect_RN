# Script Blender: Creaza un cub
# Intent: create_cube
# Parametri: size=2, location=(0,0,0)

import bpy

# Sterge obiectele existente (optional)
# bpy.ops.object.select_all(action='SELECT')
# bpy.ops.object.delete()

# Creeaza cubul
bpy.ops.mesh.primitive_cube_add(
    size=2,
    location=(0, 0, 0)
)

# Selecteaza obiectul creat
obj = bpy.context.active_object
obj.name = "Cube_Generated"
