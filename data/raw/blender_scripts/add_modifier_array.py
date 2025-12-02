# Script Blender: Adauga modifier Array
# Intent: add_modifier
# Parametri: type=array, count=5

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    array = obj.modifiers.new(name="Array", type='ARRAY')
    array.count = 5
    array.relative_offset_displace[0] = 1.1
    print("Modifier Array adaugat cu 5 copii.")
else:
    print("Selecteaza un obiect mesh!")
