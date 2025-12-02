# Script Blender: Creeaza o lumina punctiforma
# Intent: create_light
# Parametri: type=point, location=(4, 1, 6)

import bpy

# Creeaza lumina point
bpy.ops.object.light_add(
    type='POINT',
    location=(4, 1, 6)
)

light = bpy.context.active_object
light.name = "Light_Point"

# Seteaza energia luminii
light.data.energy = 1000

print("Lumina punctiforma creata.")
