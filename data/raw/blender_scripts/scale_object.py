# Script Blender: Scaleaza obiectul activ
# Intent: scale_object
# Parametri: sx=2, sy=2, sz=2

import bpy

# Obtine obiectul activ
obj = bpy.context.active_object

if obj:
    # Seteaza scala
    obj.scale = (2, 2, 2)
    print(f"Obiectul '{obj.name}' a fost scalat la (2, 2, 2)")
else:
    print("Niciun obiect selectat!")
