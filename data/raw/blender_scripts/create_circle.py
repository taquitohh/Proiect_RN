# Script Blender: Creeaza un cerc (circle/edge loop)
# Intent: create_circle
# Parametri: radius=1, vertices=32

import bpy

bpy.ops.mesh.primitive_circle_add(
    vertices=32,
    radius=1,
    location=(0, 0, 0),
    fill_type='NOTHING'
)

obj = bpy.context.active_object
obj.name = "Circle_Generated"

print("Cerc creat!")
