import json
import re
from typing import Dict, Any, List, Tuple

class BlenderScriptGenerator:
    def __init__(self):
        self.default_params = {
            "size": 2.0,
            "radius": 1.0,
            "depth": 2.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.0
        }
        
        # Template-uri implicite (fallback)
        self.templates = {
            "create_cube": "bpy.ops.mesh.primitive_cube_add(size={size}, location=({x}, {y}, {z}))",
            "create_sphere": "bpy.ops.mesh.primitive_uv_sphere_add(radius={radius}, location=({x}, {y}, {z}))",
            "create_cylinder": "bpy.ops.mesh.primitive_cylinder_add(radius={radius}, depth={depth}, location=({x}, {y}, {z}))",
            "create_cone": "bpy.ops.mesh.primitive_cone_add(radius1={radius}, depth={depth}, location=({x}, {y}, {z}))",
            "delete_all": "bpy.ops.object.select_all(action='SELECT')\nbpy.ops.object.delete()"
        }

    def generate_script(self, intent: str, extracted_params: Dict[str, Any]) -> str:
        """
        Generează scriptul final combinând intentia (clasa prezisă de AI)
        cu parametrii extrași din text.
        """
        # 1. Selectare template
        template = self.templates.get(intent, "")
        if not template:
            return "# Eroare: Nu am putut identifica un template pentru intentia: " + intent

        # 2. Combinare parametri impliciți cu cei extrași
        final_params = self.default_params.copy()
        final_params.update(extracted_params)

        # 3. Completare template
        try:
            script = "import bpy\n\n" + template.format(**final_params)
            return script
        except KeyError as e:
            return f"# Eroare generare script: Parametrul {e} lipsește din template."

    def extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extrage numere și entități simple folosind Regex (Rule-based extraction).
        Aceasta funcționează în tandem cu Rețeaua Neuronală care detectează DOAR intentia.
        """
        params = {}
        text = text.lower()

        # Extragere dimensiuni simple (ex: "raza 5", "inaltime 2")
        radius_match = re.search(r'(?:raza|radius)\s*(\d+(?:\.\d+)?)', text)
        if radius_match:
            params['radius'] = float(radius_match.group(1))

        size_match = re.search(r'(?:marime|dimensiune|latura|size)\s*(\d+(?:\.\d+)?)', text)
        if size_match:
            params['size'] = float(size_match.group(1))
            
        depth_match = re.search(r'(?:inaltime|lungime|depth|height)\s*(\d+(?:\.\d+)?)', text)
        if depth_match:
            params['depth'] = float(depth_match.group(1))

        # Extragere coordonate (ex: "pozitia 10 20 5")
        pos_match = re.search(r'(?:pozitia|locatia|la)\s*(\d+)\s+(\d+)\s+(\d+)', text)
        if pos_match:
            params['x'] = float(pos_match.group(1))
            params['y'] = float(pos_match.group(2))
            params['z'] = float(pos_match.group(3))

        return params
