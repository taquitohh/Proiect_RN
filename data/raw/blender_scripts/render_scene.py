# Script Blender: Randeaza scena
# Intent: render
# Parametri: none

import bpy

# Seteaza engine-ul de render
bpy.context.scene.render.engine = 'CYCLES'

# Seteaza rezolutia
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080

# Seteaza calea de output
bpy.context.scene.render.filepath = "//render_output.png"

# Randeaza
bpy.ops.render.render(write_still=True)

print("Render complet! Salvat in render_output.png")
