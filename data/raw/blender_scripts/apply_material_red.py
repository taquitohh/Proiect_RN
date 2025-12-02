# Script Blender: Aplica material rosu
# Intent: apply_material
# Parametri: color=red (1, 0, 0, 1)

import bpy

# Obtine obiectul activ
obj = bpy.context.active_object

if obj:
    # Creeaza un material nou
    mat = bpy.data.materials.new(name="Red_Material")
    mat.use_nodes = True
    
    # Obtine nodul Principled BSDF
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Seteaza culoarea rosie (R, G, B, Alpha)
        bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)
    
    # Aplica materialul pe obiect
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"Material rosu aplicat pe '{obj.name}'")
else:
    print("Niciun obiect selectat!")
