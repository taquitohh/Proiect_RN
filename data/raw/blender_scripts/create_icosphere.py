# Script Blender: Creeaza un icosphere
# Intent: create_icosphere
# Parametri: radius=1, subdivisions=2

import bpy

bpy.ops.mesh.primitive_ico_sphere_add(
    radius=1,
    subdivisions=2,
    location=(0, 0, 0)
)

obj = bpy.context.active_object
obj.name = "Icosphere_Generated"

print("Icosphere creata!")
