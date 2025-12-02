# Script Blender: Creeaza o piramida (con cu 4 laturi)
# Intent: create_pyramid
# Parametri: size=2

import bpy

bpy.ops.mesh.primitive_cone_add(
    vertices=4,
    radius1=2,
    radius2=0,
    depth=2,
    location=(0, 0, 1)
)

obj = bpy.context.active_object
obj.name = "Pyramid"

print("Piramida creata!")
