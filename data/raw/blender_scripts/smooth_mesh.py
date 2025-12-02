# Script Blender: Smooth mesh (shade smooth)
# Intent: smooth
# Parametri: none

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    # Aplica shade smooth
    bpy.ops.object.shade_smooth()
    print("Smooth shading aplicat.")
else:
    print("Selecteaza un obiect mesh!")
