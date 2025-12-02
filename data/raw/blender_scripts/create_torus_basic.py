# Script Blender: Creaza un torus (inel/gogoasa)
# Intent: create_torus
# Parametri: major_radius=1, minor_radius=0.25, location=(0,0,0)

import bpy

bpy.ops.mesh.primitive_torus_add(
    major_radius=1,
    minor_radius=0.25,
    location=(0, 0, 0),
    major_segments=48,
    minor_segments=12
)

obj = bpy.context.active_object
obj.name = "Torus_Generated"
