# Script Blender: Sterge toate obiectele din scena
# Intent: delete_all
# Parametri: none

import bpy

# Selecteaza toate obiectele
bpy.ops.object.select_all(action='SELECT')

# Sterge toate obiectele selectate
bpy.ops.object.delete()

print("Toate obiectele au fost sterse din scena.")
