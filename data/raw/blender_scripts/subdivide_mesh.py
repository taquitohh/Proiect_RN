# Script Blender: Subdivide mesh
# Intent: subdivide
# Parametri: cuts=1

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.subdivide(number_cuts=1)
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Mesh-ul a fost subdivivat.")
else:
    print("Selecteaza un obiect mesh!")
