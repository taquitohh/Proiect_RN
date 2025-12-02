# Script Blender: Creeaza o lumina spot
# Intent: create_light
# Parametri: type=spot

import bpy
import math

bpy.ops.object.light_add(
    type='SPOT',
    location=(5, 5, 5)
)

light = bpy.context.active_object
light.name = "Light_Spot"
light.data.energy = 1000
light.data.spot_size = math.radians(45)

# Roteste spotul in jos
light.rotation_euler = (math.radians(45), 0, math.radians(45))

print("Lumina spot creata.")
