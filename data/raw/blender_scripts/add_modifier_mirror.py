# Script Blender: Adauga modifier Mirror
# Intent: add_modifier
# Parametri: type=mirror

import bpy

obj = bpy.context.active_object

if obj and obj.type == 'MESH':
    mirror = obj.modifiers.new(name="Mirror", type='MIRROR')
    mirror.use_axis[0] = True  # Mirror pe X
    print("Modifier Mirror adaugat pe axa X.")
else:
    print("Selecteaza un obiect mesh!")
