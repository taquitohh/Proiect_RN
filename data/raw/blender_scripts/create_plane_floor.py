# Script Blender: Creaza un plan (podea)
# Intent: create_plane
# Parametri: size=10, location=(0,0,0)

import bpy

bpy.ops.mesh.primitive_plane_add(
    size=10,
    location=(0, 0, 0)
)

obj = bpy.context.active_object
obj.name = "Plane_Floor"
