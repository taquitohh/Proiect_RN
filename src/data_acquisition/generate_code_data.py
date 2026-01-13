"""
Training Data Generator for Code Generation Model

Generates pairs of (natural_language_prompt, blender_python_code)
for training the sequence-to-sequence model.

NO TEMPLATES AT RUNTIME - the model learns to generate code directly.
"""

import json
import random
import os
from typing import List, Dict, Tuple
from pathlib import Path


class BlenderCodeDataGenerator:
    """
    Generates training data for the code generation model.
    Each sample is a (prompt, code) pair.
    """
    
    def __init__(self):
        # Define all possible Blender operations and their code
        self.operations = self._define_operations()
        self.prompt_templates = self._define_prompt_templates()
        
    def _define_operations(self) -> Dict[str, Dict]:
        """Define all Blender operations with their Python code."""
        
        return {
            # ==================== PRIMITIVES ====================
            "create_cube": {
                "code": """import bpy
bpy.ops.mesh.primitive_cube_add(size={size}, location=({x}, {y}, {z}))
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "size": [1, 2, 3, 0.5, 1.5, 2.5],
                    "x": [0, 1, -1, 2, -2, 3, -3],
                    "y": [0, 1, -1, 2, -2, 3, -3],
                    "z": [0, 1, -1, 2, -2, 0.5, 1.5],
                    "name": ["Cube", "MyCube", "Box", "Block", "Cub"]
                }
            },
            
            "create_sphere": {
                "code": """import bpy
bpy.ops.mesh.primitive_uv_sphere_add(radius={radius}, location=({x}, {y}, {z}), segments={segments}, ring_count={rings})
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "radius": [1, 0.5, 2, 1.5, 0.75],
                    "x": [0, 1, -1, 2, -2],
                    "y": [0, 1, -1, 2, -2],
                    "z": [0, 1, 2, 0.5, 1.5],
                    "segments": [32, 16, 24, 48],
                    "rings": [16, 12, 24, 32],
                    "name": ["Sphere", "Ball", "Sfera", "MySphere"]
                }
            },
            
            "create_cylinder": {
                "code": """import bpy
bpy.ops.mesh.primitive_cylinder_add(radius={radius}, depth={depth}, location=({x}, {y}, {z}), vertices={vertices})
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "radius": [1, 0.5, 2, 0.75, 1.5],
                    "depth": [2, 1, 3, 4, 1.5],
                    "x": [0, 1, -1, 2],
                    "y": [0, 1, -1, 2],
                    "z": [0, 1, 0.5, 1.5],
                    "vertices": [32, 16, 24, 48],
                    "name": ["Cylinder", "Tube", "Cilindru", "Pillar"]
                }
            },
            
            "create_cone": {
                "code": """import bpy
bpy.ops.mesh.primitive_cone_add(radius1={radius1}, radius2={radius2}, depth={depth}, location=({x}, {y}, {z}))
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "radius1": [1, 0.5, 2, 1.5],
                    "radius2": [0, 0.25, 0.5],
                    "depth": [2, 1, 3, 1.5],
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0, 1, 0.5],
                    "name": ["Cone", "Con", "Pyramid", "Spike"]
                }
            },
            
            "create_torus": {
                "code": """import bpy
bpy.ops.mesh.primitive_torus_add(major_radius={major_radius}, minor_radius={minor_radius}, location=({x}, {y}, {z}))
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "major_radius": [1, 1.5, 2, 0.75],
                    "minor_radius": [0.25, 0.5, 0.3, 0.1],
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0, 1, 0.5],
                    "name": ["Torus", "Donut", "Ring", "Tor"]
                }
            },
            
            "create_plane": {
                "code": """import bpy
bpy.ops.mesh.primitive_plane_add(size={size}, location=({x}, {y}, {z}))
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "size": [2, 1, 4, 5, 10],
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0, -1, -0.5],
                    "name": ["Plane", "Floor", "Ground", "Platform"]
                }
            },
            
            "create_icosphere": {
                "code": """import bpy
bpy.ops.mesh.primitive_ico_sphere_add(radius={radius}, subdivisions={subdivisions}, location=({x}, {y}, {z}))
obj = bpy.context.active_object
obj.name = "{name}"
""",
                "params": {
                    "radius": [1, 0.5, 2, 1.5],
                    "subdivisions": [2, 3, 4, 1],
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0, 1, 0.5],
                    "name": ["IcoSphere", "GeoSphere", "Icosahedru"]
                }
            },
            
            # ==================== MATERIALS ====================
            "apply_material": {
                "code": """import bpy
obj = bpy.context.active_object
mat = bpy.data.materials.new(name="{mat_name}")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)
bsdf.inputs["Metallic"].default_value = {metallic}
bsdf.inputs["Roughness"].default_value = {roughness}
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
""",
                "params": {
                    "mat_name": ["Material", "MyMaterial", "Color"],
                    "r": [1.0, 0.8, 0.0, 0.0, 0.5, 0.2, 1.0],
                    "g": [0.0, 0.2, 0.8, 0.0, 0.5, 0.2, 1.0],
                    "b": [0.0, 0.2, 0.0, 1.0, 0.5, 0.2, 1.0],
                    "metallic": [0.0, 0.5, 1.0, 0.8],
                    "roughness": [0.5, 0.2, 0.8, 0.1, 1.0]
                },
                "color_presets": {
                    "red": (1.0, 0.0, 0.0),
                    "rosu": (1.0, 0.0, 0.0),
                    "green": (0.0, 0.8, 0.0),
                    "verde": (0.0, 0.8, 0.0),
                    "blue": (0.0, 0.0, 1.0),
                    "albastru": (0.0, 0.0, 1.0),
                    "yellow": (1.0, 1.0, 0.0),
                    "galben": (1.0, 1.0, 0.0),
                    "white": (1.0, 1.0, 1.0),
                    "alb": (1.0, 1.0, 1.0),
                    "black": (0.0, 0.0, 0.0),
                    "negru": (0.0, 0.0, 0.0),
                    "orange": (1.0, 0.5, 0.0),
                    "portocaliu": (1.0, 0.5, 0.0),
                    "purple": (0.5, 0.0, 0.5),
                    "mov": (0.5, 0.0, 0.5),
                    "pink": (1.0, 0.4, 0.7),
                    "roz": (1.0, 0.4, 0.7),
                    "gray": (0.5, 0.5, 0.5),
                    "gri": (0.5, 0.5, 0.5),
                    "gold": (1.0, 0.84, 0.0),
                    "auriu": (1.0, 0.84, 0.0),
                    "silver": (0.75, 0.75, 0.75),
                    "argintiu": (0.75, 0.75, 0.75)
                }
            },
            
            "apply_glass_material": {
                "code": """import bpy
obj = bpy.context.active_object
mat = bpy.data.materials.new(name="Glass")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)
bsdf.inputs["Transmission"].default_value = {transmission}
bsdf.inputs["Roughness"].default_value = {roughness}
bsdf.inputs["IOR"].default_value = {ior}
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
""",
                "params": {
                    "r": [1.0, 0.9, 0.8],
                    "g": [1.0, 0.9, 0.8],
                    "b": [1.0, 0.9, 0.8],
                    "transmission": [1.0, 0.9, 0.8],
                    "roughness": [0.0, 0.1, 0.05],
                    "ior": [1.45, 1.5, 1.33, 2.0]
                }
            },
            
            "apply_metal_material": {
                "code": """import bpy
obj = bpy.context.active_object
mat = bpy.data.materials.new(name="Metal")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = {roughness}
if obj.data.materials:
    obj.data.materials[0] = mat
else:
    obj.data.materials.append(mat)
""",
                "params": {
                    "r": [0.8, 0.9, 0.7, 1.0],
                    "g": [0.8, 0.7, 0.6, 0.84],
                    "b": [0.8, 0.6, 0.5, 0.0],
                    "roughness": [0.2, 0.1, 0.3, 0.4]
                }
            },
            
            # ==================== MODIFIERS ====================
            "add_subdivision": {
                "code": """import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
mod.levels = {levels}
mod.render_levels = {render_levels}
""",
                "params": {
                    "levels": [1, 2, 3],
                    "render_levels": [2, 3, 4]
                }
            },
            
            "add_bevel": {
                "code": """import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Bevel", type='BEVEL')
mod.width = {width}
mod.segments = {segments}
""",
                "params": {
                    "width": [0.02, 0.05, 0.1, 0.03],
                    "segments": [2, 3, 4, 5]
                }
            },
            
            "add_mirror": {
                "code": """import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Mirror", type='MIRROR')
mod.use_axis[0] = {use_x}
mod.use_axis[1] = {use_y}
mod.use_axis[2] = {use_z}
""",
                "params": {
                    "use_x": [True, False],
                    "use_y": [False, True],
                    "use_z": [False]
                }
            },
            
            "add_array": {
                "code": """import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Array", type='ARRAY')
mod.count = {count}
mod.relative_offset_displace[0] = {offset_x}
mod.relative_offset_displace[1] = {offset_y}
mod.relative_offset_displace[2] = {offset_z}
""",
                "params": {
                    "count": [2, 3, 4, 5, 10],
                    "offset_x": [1.0, 1.5, 2.0, 0],
                    "offset_y": [0, 1.0, 1.5],
                    "offset_z": [0, 0, 1.0]
                }
            },
            
            "add_solidify": {
                "code": """import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
mod.thickness = {thickness}
mod.offset = {offset}
""",
                "params": {
                    "thickness": [0.1, 0.2, 0.05, 0.5],
                    "offset": [1, 0, -1]
                }
            },
            
            # ==================== TRANSFORMATIONS ====================
            "move_object": {
                "code": """import bpy
obj = bpy.context.active_object
obj.location = ({x}, {y}, {z})
""",
                "params": {
                    "x": [0, 1, 2, 3, -1, -2, 5],
                    "y": [0, 1, 2, 3, -1, -2, 5],
                    "z": [0, 1, 2, 3, 0.5, -1]
                }
            },
            
            "rotate_object": {
                "code": """import bpy
import math
obj = bpy.context.active_object
obj.rotation_euler = (math.radians({x}), math.radians({y}), math.radians({z}))
""",
                "params": {
                    "x": [0, 45, 90, 180, 30, 60],
                    "y": [0, 45, 90, 180, 30, 60],
                    "z": [0, 45, 90, 180, 30, 60]
                }
            },
            
            "scale_object": {
                "code": """import bpy
obj = bpy.context.active_object
obj.scale = ({x}, {y}, {z})
""",
                "params": {
                    "x": [1, 2, 0.5, 1.5, 3],
                    "y": [1, 2, 0.5, 1.5, 3],
                    "z": [1, 2, 0.5, 1.5, 3]
                }
            },
            
            # ==================== LIGHTING ====================
            "create_point_light": {
                "code": """import bpy
bpy.ops.object.light_add(type='POINT', location=({x}, {y}, {z}))
light = bpy.context.active_object
light.data.energy = {energy}
light.data.color = ({r}, {g}, {b})
light.name = "{name}"
""",
                "params": {
                    "x": [0, 3, -3, 5],
                    "y": [0, 3, -3, 5],
                    "z": [3, 5, 7, 10],
                    "energy": [100, 500, 1000, 200],
                    "r": [1.0],
                    "g": [1.0],
                    "b": [1.0],
                    "name": ["PointLight", "Light", "Lamp"]
                }
            },
            
            "create_sun_light": {
                "code": """import bpy
bpy.ops.object.light_add(type='SUN', location=({x}, {y}, {z}))
light = bpy.context.active_object
light.data.energy = {energy}
light.name = "{name}"
""",
                "params": {
                    "x": [0, 5],
                    "y": [0, 5],
                    "z": [10, 15, 20],
                    "energy": [1, 2, 3, 5],
                    "name": ["Sun", "SunLight", "Soare"]
                }
            },
            
            "create_area_light": {
                "code": """import bpy
bpy.ops.object.light_add(type='AREA', location=({x}, {y}, {z}))
light = bpy.context.active_object
light.data.energy = {energy}
light.data.size = {size}
light.name = "{name}"
""",
                "params": {
                    "x": [0, 3, -3],
                    "y": [0, 3, -3],
                    "z": [3, 5, 7],
                    "energy": [100, 200, 500],
                    "size": [1, 2, 3, 5],
                    "name": ["AreaLight", "SoftLight"]
                }
            },
            
            # ==================== CAMERA ====================
            "create_camera": {
                "code": """import bpy
bpy.ops.object.camera_add(location=({x}, {y}, {z}))
camera = bpy.context.active_object
camera.name = "{name}"
camera.rotation_euler = (math.radians({rot_x}), math.radians({rot_y}), math.radians({rot_z}))
bpy.context.scene.camera = camera
""",
                "params": {
                    "x": [7, 10, 5, -5],
                    "y": [-7, -10, 5, -5],
                    "z": [5, 7, 3, 10],
                    "rot_x": [60, 70, 80, 45],
                    "rot_y": [0],
                    "rot_z": [45, 135, 90, -45],
                    "name": ["Camera", "MainCamera", "RenderCam"]
                }
            },
            
            # ==================== SCENE OPERATIONS ====================
            "delete_all": {
                "code": """import bpy
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
""",
                "params": {}
            },
            
            "delete_selected": {
                "code": """import bpy
bpy.ops.object.delete()
""",
                "params": {}
            },
            
            "render_image": {
                "code": """import bpy
bpy.context.scene.render.resolution_x = {width}
bpy.context.scene.render.resolution_y = {height}
bpy.context.scene.render.filepath = "{filepath}"
bpy.ops.render.render(write_still=True)
""",
                "params": {
                    "width": [1920, 1280, 1024],
                    "height": [1080, 720, 768],
                    "filepath": ["/tmp/render.png", "//render.png"]
                }
            },
            
            # ==================== TEXT ====================
            "create_text": {
                "code": """import bpy
bpy.ops.object.text_add(location=({x}, {y}, {z}))
text_obj = bpy.context.active_object
text_obj.data.body = "{text}"
text_obj.data.size = {size}
text_obj.name = "{name}"
""",
                "params": {
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0, 1, 0.5],
                    "text": ["Hello", "Blender", "Text", "3D"],
                    "size": [1, 2, 0.5, 1.5],
                    "name": ["Text", "Label", "Title"]
                }
            },
            
            # ==================== CURVES ====================
            "create_bezier_curve": {
                "code": """import bpy
bpy.ops.curve.primitive_bezier_curve_add(location=({x}, {y}, {z}))
curve = bpy.context.active_object
curve.name = "{name}"
""",
                "params": {
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0],
                    "name": ["BezierCurve", "Curve", "Path"]
                }
            },
            
            "create_circle_curve": {
                "code": """import bpy
bpy.ops.curve.primitive_bezier_circle_add(radius={radius}, location=({x}, {y}, {z}))
circle = bpy.context.active_object
circle.name = "{name}"
""",
                "params": {
                    "radius": [1, 2, 0.5, 1.5],
                    "x": [0, 1, -1],
                    "y": [0, 1, -1],
                    "z": [0],
                    "name": ["Circle", "Ring", "Cerc"]
                }
            }
        }
    
    def _define_prompt_templates(self) -> Dict[str, List[str]]:
        """Define prompt templates for each operation."""
        
        return {
            "create_cube": [
                "create a cube",
                "make a cube",
                "add a cube",
                "cub",
                "cube",
                "creaza un cub",
                "fa un cub",
                "adauga un cub",
                "vreau un cub",
                "creeaza cub",
                "pune un cub",
                "create cube at position {x} {y} {z}",
                "make a cube with size {size}",
                "add a large cube",
                "create a small cube",
                "box",
                "create a box"
            ],
            
            "create_sphere": [
                "create a sphere",
                "make a sphere",
                "add a sphere",
                "sfera",
                "sphere",
                "creaza o sfera",
                "fa o sfera",
                "adauga o sfera",
                "vreau o sfera",
                "ball",
                "create a ball",
                "minge",
                "create sphere with radius {radius}"
            ],
            
            "create_cylinder": [
                "create a cylinder",
                "make a cylinder",
                "add a cylinder",
                "cilindru",
                "cylinder",
                "creaza un cilindru",
                "fa un cilindru",
                "tub",
                "tube",
                "pillar",
                "coloana"
            ],
            
            "create_cone": [
                "create a cone",
                "make a cone",
                "add a cone",
                "con",
                "cone",
                "creaza un con",
                "fa un con",
                "pyramid",
                "piramida"
            ],
            
            "create_torus": [
                "create a torus",
                "make a torus",
                "add a torus",
                "tor",
                "torus",
                "donut",
                "gogoasa",
                "creaza un tor",
                "ring shape"
            ],
            
            "create_plane": [
                "create a plane",
                "make a plane",
                "add a plane",
                "plan",
                "plane",
                "floor",
                "podea",
                "ground",
                "suprafata",
                "creaza un plan"
            ],
            
            "create_icosphere": [
                "create an icosphere",
                "make an icosphere",
                "icosphere",
                "icosfera",
                "geo sphere",
                "geodesic sphere"
            ],
            
            "apply_material": [
                "apply {color} material",
                "make it {color}",
                "color it {color}",
                "{color} material",
                "material {color}",
                "aplica material {color}",
                "fa-l {color}",
                "coloreaza {color}",
                "pune culoare {color}",
                "add {color} color"
            ],
            
            "apply_glass_material": [
                "apply glass material",
                "make it glass",
                "transparent material",
                "sticla",
                "material sticla",
                "glass",
                "transparent"
            ],
            
            "apply_metal_material": [
                "apply metal material",
                "make it metallic",
                "metal",
                "metalic",
                "material metal",
                "shiny metal"
            ],
            
            "add_subdivision": [
                "add subdivision",
                "subdivide",
                "smooth mesh",
                "subdivision surface",
                "subsurf",
                "make it smooth"
            ],
            
            "add_bevel": [
                "add bevel",
                "bevel edges",
                "round edges",
                "bevel",
                "rotunjeste muchiile"
            ],
            
            "add_mirror": [
                "add mirror",
                "mirror modifier",
                "mirror",
                "oglinda",
                "simetrie"
            ],
            
            "add_array": [
                "add array",
                "array modifier",
                "duplicate in array",
                "multiply object",
                "array"
            ],
            
            "add_solidify": [
                "add solidify",
                "add thickness",
                "solidify",
                "make solid"
            ],
            
            "move_object": [
                "move to {x} {y} {z}",
                "move object",
                "translate",
                "muta la {x} {y} {z}",
                "muta obiectul",
                "position at {x} {y} {z}"
            ],
            
            "rotate_object": [
                "rotate {x} degrees",
                "rotate object",
                "rotation",
                "roteste",
                "roteste cu {x} grade"
            ],
            
            "scale_object": [
                "scale to {x}",
                "scale object",
                "resize",
                "mareste",
                "micsoreaza",
                "make bigger",
                "make smaller"
            ],
            
            "create_point_light": [
                "add point light",
                "create light",
                "point light",
                "lumina punct",
                "adauga lumina",
                "lamp"
            ],
            
            "create_sun_light": [
                "add sun light",
                "create sun",
                "sun light",
                "soare",
                "lumina soare"
            ],
            
            "create_area_light": [
                "add area light",
                "area light",
                "soft light",
                "lumina zona"
            ],
            
            "create_camera": [
                "add camera",
                "create camera",
                "camera",
                "adauga camera",
                "creaza camera"
            ],
            
            "delete_all": [
                "delete all",
                "clear scene",
                "remove everything",
                "sterge tot",
                "curata scena",
                "delete all objects"
            ],
            
            "delete_selected": [
                "delete",
                "delete selected",
                "remove object",
                "sterge",
                "sterge obiectul"
            ],
            
            "render_image": [
                "render",
                "render image",
                "create render",
                "randeaza",
                "render scene"
            ],
            
            "create_text": [
                "add text",
                "create text",
                "text object",
                "adauga text",
                "scrie text"
            ],
            
            "create_bezier_curve": [
                "add bezier curve",
                "create curve",
                "bezier",
                "curba bezier"
            ],
            
            "create_circle_curve": [
                "add circle",
                "create circle",
                "circle curve",
                "cerc",
                "adauga cerc"
            ]
        }
    
    def generate_sample(self, operation: str) -> Tuple[str, str]:
        """Generate a single (prompt, code) sample."""
        
        op_data = self.operations[operation]
        templates = self.prompt_templates.get(operation, [operation])
        
        # Choose a random prompt template
        prompt_template = random.choice(templates)
        
        # Generate random parameters
        params = {}
        for param_name, param_values in op_data.get("params", {}).items():
            params[param_name] = random.choice(param_values)
        
        # Handle color presets for materials
        if operation == "apply_material" and "{color}" in prompt_template:
            color_presets = op_data.get("color_presets", {})
            color_name = random.choice(list(color_presets.keys()))
            r, g, b = color_presets[color_name]
            params["r"] = r
            params["g"] = g
            params["b"] = b
            prompt_template = prompt_template.replace("{color}", color_name)
        
        # Fill in prompt template
        prompt = prompt_template
        for param_name, value in params.items():
            prompt = prompt.replace("{" + param_name + "}", str(value))
        
        # Generate code
        code = op_data["code"].format(**params)
        
        return prompt, code
    
    def generate_dataset(self, samples_per_operation: int = 50) -> List[Dict]:
        """Generate full training dataset."""
        
        dataset = []
        
        for operation in self.operations:
            for _ in range(samples_per_operation):
                prompt, code = self.generate_sample(operation)
                dataset.append({
                    "prompt": prompt,
                    "code": code,
                    "operation": operation
                })
        
        # Shuffle
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(self, output_path: str, samples_per_operation: int = 50):
        """Generate and save dataset to JSON."""
        
        dataset = self.generate_dataset(samples_per_operation)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Generated {len(dataset)} samples")
        print(f"Saved to {output_path}")
        
        # Stats
        ops_count = {}
        for sample in dataset:
            op = sample["operation"]
            ops_count[op] = ops_count.get(op, 0) + 1
        
        print("\nSamples per operation:")
        for op, count in sorted(ops_count.items()):
            print(f"  {op}: {count}")
        
        return dataset


def main():
    """Generate training data for code generation model."""
    
    generator = BlenderCodeDataGenerator()
    
    output_path = "data_red/generated/code_generation_data.json"
    dataset = generator.save_dataset(output_path, samples_per_operation=100)
    
    # Show example
    print("\n" + "="*60)
    print("EXAMPLE SAMPLES:")
    print("="*60)
    
    for sample in random.sample(dataset, 3):
        print(f"\nPrompt: {sample['prompt']}")
        print(f"Operation: {sample['operation']}")
        print(f"Code:\n{sample['code']}")
        print("-" * 40)


if __name__ == "__main__":
    main()
