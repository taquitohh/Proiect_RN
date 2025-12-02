# Script Blender: Aplica material metalic
# Intent: apply_material
# Parametri: material=metal

import bpy

obj = bpy.context.active_object

if obj:
    mat = bpy.data.materials.new(name="Metal_Material")
    mat.use_nodes = True
    
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)
        bsdf.inputs['Metallic'].default_value = 1.0
        bsdf.inputs['Roughness'].default_value = 0.2
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"Material metalic aplicat pe '{obj.name}'")
else:
    print("Niciun obiect selectat!")
