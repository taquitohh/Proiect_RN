# Script Blender: Roteste obiectul activ
# Intent: rotate_object
# Parametri: rx=0, ry=0, rz=45 (grade)

import bpy
import math

# Obtine obiectul activ
obj = bpy.context.active_object

if obj:
    # Converteste gradele in radiani
    rx = math.radians(0)
    ry = math.radians(0)
    rz = math.radians(45)
    
    # Seteaza rotatia
    obj.rotation_euler = (rx, ry, rz)
    print(f"Obiectul '{obj.name}' a fost rotit cu 45 grade pe axa Z")
else:
    print("Niciun obiect selectat!")
