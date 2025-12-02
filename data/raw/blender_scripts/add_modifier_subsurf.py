# Script Blender: Adauga modifier Subdivision Surface
# Intent: add_modifier
# Parametri: type=subsurf

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.levels = 2
    subsurf.render_levels = 3
    print("Modifier Subdivision Surface adaugat.")
else:
    print("Selecteaza un obiect mesh!")
