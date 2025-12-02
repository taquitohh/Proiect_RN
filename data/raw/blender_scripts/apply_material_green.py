# Script Blender: Aplica material verde
# Intent: apply_material
# Parametri: color=green (0, 1, 0, 1)

import bpy

obj = bpy.context.active_object

if obj:
    mat = bpy.data.materials.new(name="Green_Material")
    mat.use_nodes = True
    
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0, 1, 0, 1)
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"Material verde aplicat pe '{obj.name}'")
else:
    print("Niciun obiect selectat!")
