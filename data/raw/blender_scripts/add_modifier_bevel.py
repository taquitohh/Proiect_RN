# Script Blender: Adauga modifier Bevel
# Intent: add_modifier
# Parametri: type=bevel

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    # Adauga modifier bevel
    bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = 0.05
    bevel.segments = 3
    print("Modifier Bevel adaugat.")
else:
    print("Selecteaza un obiect mesh!")
