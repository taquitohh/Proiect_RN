# Script Blender: Creeaza o lumina directionala (soare)
# Intent: create_light
# Parametri: type=sun

import bpy

bpy.ops.object.light_add(
    type='SUN',
    location=(0, 0, 10)
)

light = bpy.context.active_object
light.name = "Light_Sun"
light.data.energy = 5

print("Lumina directionala (soare) creata.")
