# Script Blender: Duplica obiectul activ
# Intent: duplicate_object
# Parametri: none

import bpy

# Duplica obiectul selectat
bpy.ops.object.duplicate(linked=False)

# Muta duplicatul putin pentru vizibilitate
obj = bpy.context.active_object
if obj:
    obj.location.x += 2
    print(f"Obiectul a fost duplicat: '{obj.name}'")
