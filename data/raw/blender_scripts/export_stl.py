# Script Blender: Exporta ca STL (pentru printare 3D)
# Intent: export_file
# Parametri: format=stl

import bpy

filepath = "//export.stl"

bpy.ops.wm.stl_export(filepath=bpy.path.abspath(filepath))

print(f"Scena exportata ca STL: {filepath}")
