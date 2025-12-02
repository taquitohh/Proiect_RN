# Script Blender: Creaza un cilindru
# Intent: create_cylinder
# Parametri: radius=1, depth=2, location=(0,0,0)

import bpy

bpy.ops.mesh.primitive_cylinder_add(
    radius=1,
    depth=2,
    location=(0, 0, 0),
    vertices=32
)

obj = bpy.context.active_object
obj.name = "Cylinder_Generated"
