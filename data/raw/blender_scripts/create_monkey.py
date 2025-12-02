# Script Blender: Creeaza o maimuta Suzanne
# Intent: create_monkey
# Parametri: none

import bpy

bpy.ops.mesh.primitive_monkey_add(
    size=2,
    location=(0, 0, 0)
)

obj = bpy.context.active_object
obj.name = "Suzanne"

# Aplica smooth shading
bpy.ops.object.shade_smooth()

print("Suzanne creata!")
