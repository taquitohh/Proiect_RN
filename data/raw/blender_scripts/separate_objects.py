# Script Blender: Separa obiectele (by loose parts)
# Intent: separate_objects
# Parametri: none

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    # Intra in edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    # Separa mesh-ul
    bpy.ops.mesh.separate(type='LOOSE')
    # Inapoi in object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Mesh-ul a fost separat.")
else:
    print("Selecteaza un obiect mesh!")
