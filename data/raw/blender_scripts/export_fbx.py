# Script Blender: Exporta ca FBX
# Intent: export_file
# Parametri: format=fbx

import bpy

# Seteaza calea de export
filepath = "//export.fbx"

# Exporta in format FBX
bpy.ops.export_scene.fbx(filepath=bpy.path.abspath(filepath))

print(f"Scena exportata ca FBX: {filepath}")
