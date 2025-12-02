# Script Blender: Creaza o sfera mare
# Intent: create_sphere
# Parametri: radius=3, location=(1,1,1)

import bpy

bpy.ops.mesh.primitive_uv_sphere_add(
    radius=3,
    location=(1, 1, 1),
    segments=64,
    ring_count=32
)

obj = bpy.context.active_object
obj.name = "Sphere_Large"
