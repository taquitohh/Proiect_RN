# Script Blender: Creeaza o scena completa (cub + lumina + camera)
# Intent: create_scene
# Parametri: none

import bpy

# Sterge tot
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Creeaza cub
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
cube = bpy.context.active_object
cube.name = "Main_Cube"

# Creeaza podea
bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
plane = bpy.context.active_object
plane.name = "Floor"

# Creeaza lumina
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
light = bpy.context.active_object
light.name = "Sun_Light"
light.data.energy = 5

# Creeaza camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.active_object
camera.name = "Main_Camera"
camera.rotation_euler = (1.1, 0, 0.78)
bpy.context.scene.camera = camera

print("Scena completa creata!")
