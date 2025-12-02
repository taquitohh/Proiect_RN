# Script Blender: Exporta ca OBJ
# Intent: export_file
# Parametri: format=obj

import bpy

filepath = "//export.obj"

bpy.ops.wm.obj_export(filepath=bpy.path.abspath(filepath))

print(f"Scena exportata ca OBJ: {filepath}")
