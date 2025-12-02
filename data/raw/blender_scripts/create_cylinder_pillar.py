# Script Blender: Creaza un cilindru inalt (stalp)
# Intent: create_cylinder
# Parametri: radius=0.3, depth=5, location=(0,0,2.5)

import bpy

bpy.ops.mesh.primitive_cylinder_add(
    radius=0.3,
    depth=5,
    location=(0, 0, 2.5),
    vertices=32
)

obj = bpy.context.active_object
obj.name = "Cylinder_Pillar"
