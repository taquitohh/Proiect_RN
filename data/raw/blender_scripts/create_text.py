# Script Blender: Creeaza text 3D
# Intent: create_text
# Parametri: text="Hello"

import bpy

# Creeaza text
bpy.ops.object.text_add(location=(0, 0, 0))

text_obj = bpy.context.active_object
text_obj.data.body = "Hello"
text_obj.data.size = 1
text_obj.data.extrude = 0.1

text_obj.name = "Text_3D"

print("Text 3D creat: 'Hello'")
