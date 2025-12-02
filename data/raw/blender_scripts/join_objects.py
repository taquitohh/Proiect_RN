# Script Blender: Uneste obiectele selectate
# Intent: join_objects
# Parametri: none

import bpy

# Verifica daca sunt obiecte selectate
if len(bpy.context.selected_objects) > 1:
    # Uneste obiectele
    bpy.ops.object.join()
    print("Obiectele au fost unite.")
else:
    print("Selecteaza cel putin 2 obiecte pentru a le uni!")
