# Script Blender: Creaza o sfera
# Intent: create_sphere
# Parametri: radius=1, location=(0,0,0)

import bpy

bpy.ops.mesh.primitive_uv_sphere_add(
    radius=1,
    location=(0, 0, 0),
    segments=32,
    ring_count=16
)

obj = bpy.context.active_object
obj.name = "Sphere_Generated"
