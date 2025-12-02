# Script Blender: Creaza un con inalt
# Intent: create_cone
# Parametri: radius1=1, depth=4, location=(0,0,2)

import bpy

bpy.ops.mesh.primitive_cone_add(
    radius1=1,
    radius2=0,
    depth=4,
    location=(0, 0, 2),
    vertices=32
)

obj = bpy.context.active_object
obj.name = "Cone_Tall"
