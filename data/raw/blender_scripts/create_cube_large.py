# Script Blender: Creaza un cub cu parametri custom
# Intent: create_cube
# Parametri: size=4, location=(5,5,0)

import bpy

bpy.ops.mesh.primitive_cube_add(
    size=4,
    location=(5, 5, 0)
)

obj = bpy.context.active_object
obj.name = "Cube_Large"
