# Script Blender: Creaza un con
# Intent: create_cone
# Parametri: radius1=1, depth=2, location=(0,0,0)

import bpy

bpy.ops.mesh.primitive_cone_add(
    radius1=1,
    radius2=0,
    depth=2,
    location=(0, 0, 0),
    vertices=32
)

obj = bpy.context.active_object
obj.name = "Cone_Generated"
