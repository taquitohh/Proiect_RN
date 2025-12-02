# Script Blender: Muta obiectul activ la o pozitie noua
# Intent: move_object
# Parametri: x=1, y=2, z=3

import bpy

# Obtine obiectul activ
obj = bpy.context.active_object

if obj:
    # Seteaza noua locatie
    obj.location = (1, 2, 3)
    print(f"Obiectul '{obj.name}' a fost mutat la (1, 2, 3)")
else:
    print("Niciun obiect selectat!")
