# Script Blender: Creeaza o camera
# Intent: create_camera
# Parametri: location=(7, -7, 5)

import bpy

# Creeaza camera
bpy.ops.object.camera_add(
    location=(7, -7, 5)
)

# Obtine camera creata
camera = bpy.context.active_object
camera.name = "Camera_Generated"

# Roteste camera spre origine
camera.rotation_euler = (1.1, 0, 0.78)

# Seteaza ca camera activa
bpy.context.scene.camera = camera

print("Camera creata si setata ca activa.")
