# Script Blender: Aplica material sticla
# Intent: apply_material
# Parametri: material=glass

import bpy

obj = bpy.context.active_object

if obj:
    mat = bpy.data.materials.new(name="Glass_Material")
    mat.use_nodes = True
    
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.8, 0.9, 1.0, 1)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.0
        bsdf.inputs['Transmission'].default_value = 1.0
        bsdf.inputs['IOR'].default_value = 1.45
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"Material sticla aplicat pe '{obj.name}'")
else:
    print("Niciun obiect selectat!")
