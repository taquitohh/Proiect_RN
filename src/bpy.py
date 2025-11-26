"""
Mock module pentru Blender Python API (bpy).
=============================================

Acest modul simulează comportamentul API-ului Blender pentru a permite
testarea scripturilor fără a avea Blender instalat.

NOTĂ: Acest modul este DOAR pentru testare/dezvoltare.
Scripturile reale trebuie rulate în Blender.
"""

import math
from typing import Any, Dict, List, Optional, Tuple


class MockObject:
    """Simulează un obiect Blender."""
    
    def __init__(self, name: str = "Object", obj_type: str = "MESH"):
        self.name = name
        self.type = obj_type
        self.location = [0.0, 0.0, 0.0]
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.data: Any = MockMeshData()
        self.modifiers = MockModifiers()
        
    def __repr__(self):
        return f"<MockObject '{self.name}' type='{self.type}'>"


class MockMeshData:
    """Simulează datele unui mesh."""
    
    def __init__(self):
        self.materials = []


class MockModifiers:
    """Simulează colecția de modifiers."""
    
    def __init__(self):
        self._modifiers: Dict[str, Any] = {}
        
    def new(self, name: str, type: str) -> 'MockModifier':
        mod = MockModifier(name, type)
        self._modifiers[name] = mod
        print(f"[MOCK] Modifier '{name}' ({type}) adăugat.")
        return mod


class MockModifier:
    """Simulează un modifier."""
    
    def __init__(self, name: str, mod_type: str):
        self.name = name
        self.type = mod_type
        # Proprietăți comune
        self.width = 0.05
        self.segments = 1
        self.count = 1
        self.levels = 1
        self.render_levels = 2
        self.use_axis = [True, False, False]
        self.relative_offset_displace = [1.0, 0.0, 0.0]


class MockMaterial:
    """Simulează un material Blender."""
    
    def __init__(self, name: str = "Material"):
        self.name = name
        self.use_nodes = False
        self.node_tree = MockNodeTree()


class MockNodeTree:
    """Simulează node tree pentru materiale."""
    
    def __init__(self):
        self.nodes = MockNodes()


class MockNodes:
    """Simulează colecția de nodes."""
    
    def get(self, name: str) -> Optional['MockBSDFNode']:
        if name == "Principled BSDF":
            return MockBSDFNode()
        return None


class MockBSDFNode:
    """Simulează nodul Principled BSDF."""
    
    def __init__(self):
        self.inputs = {
            'Base Color': MockInput((0.8, 0.8, 0.8, 1.0)),
            'Metallic': MockInput(0.0),
            'Roughness': MockInput(0.5),
            'Transmission': MockInput(0.0),
            'IOR': MockInput(1.45),
        }


class MockInput:
    """Simulează un input de nod."""
    
    def __init__(self, default):
        self.default_value = default


class MockLight:
    """Simulează datele unei lumini."""
    
    def __init__(self):
        self.energy = 1000
        self.spot_size = 0.785  # 45 grade în radiani


class MockCamera:
    """Simulează datele unei camere."""
    pass


class MockScene:
    """Simulează o scenă Blender."""
    
    def __init__(self):
        self.camera = None
        self.render = MockRenderSettings()


class MockRenderSettings:
    """Simulează setările de render."""
    
    def __init__(self):
        self.engine = 'CYCLES'
        self.resolution_x = 1920
        self.resolution_y = 1080
        self.filepath = ""


class MockContext:
    """Simulează bpy.context."""
    
    def __init__(self):
        self._active_object: Optional[MockObject] = None
        self._selected_objects: List[MockObject] = []
        self.scene = MockScene()
        
    @property
    def active_object(self) -> Optional[MockObject]:
        return self._active_object
    
    @property
    def selected_objects(self) -> List[MockObject]:
        return self._selected_objects


class MockData:
    """Simulează bpy.data."""
    
    def __init__(self):
        self._materials: Dict[str, MockMaterial] = {}
        self._objects: Dict[str, MockObject] = {}
        
    @property
    def materials(self):
        return self
    
    @property
    def objects(self):
        return self
        
    def new(self, name: str) -> MockMaterial:
        mat = MockMaterial(name)
        self._materials[name] = mat
        print(f"[MOCK] Material '{name}' creat.")
        return mat


class MockOps:
    """Simulează bpy.ops."""
    
    def __init__(self, context: MockContext):
        self._context = context
        self.mesh = MockMeshOps(context)
        self.object = MockObjectOps(context)
        self.wm = MockWMOps()
        self.render = MockRenderOps()
        self.export_scene = MockExportOps()


class MockMeshOps:
    """Simulează bpy.ops.mesh."""
    
    def __init__(self, context: MockContext):
        self._context = context
        
    def primitive_cube_add(self, size: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Cube", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Cub creat: size={size}, location={location}")
        
    def primitive_uv_sphere_add(self, radius: float = 1, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Sphere", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Sferă creată: radius={radius}, location={location}")
        
    def primitive_cylinder_add(self, radius: float = 1, depth: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Cylinder", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Cilindru creat: radius={radius}, depth={depth}, location={location}")
        
    def primitive_cone_add(self, radius1: float = 1, depth: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Cone", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Con creat: radius1={radius1}, depth={depth}, location={location}")
        
    def primitive_torus_add(self, major_radius: float = 1, minor_radius: float = 0.25, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Torus", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Torus creat: major_radius={major_radius}, minor_radius={minor_radius}")
        
    def primitive_plane_add(self, size: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Plane", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Plan creat: size={size}, location={location}")
        
    def primitive_grid_add(self, x_subdivisions: int = 10, y_subdivisions: int = 10, size: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Grid", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Grid creat: size={size}")
        
    def primitive_monkey_add(self, size: float = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Suzanne", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Suzanne creată: size={size}")
        
    def primitive_circle_add(self, vertices: int = 32, radius: float = 1, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Circle", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Cerc creat: radius={radius}, vertices={vertices}")
        
    def primitive_ico_sphere_add(self, radius: float = 1, subdivisions: int = 2, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Icosphere", "MESH")
        obj.location = list(location)
        self._context._active_object = obj
        print(f"[MOCK] Icosphere creată: radius={radius}")
        
    def select_all(self, action: str = 'SELECT'):
        print(f"[MOCK] mesh.select_all(action='{action}')")
        
    def subdivide(self, number_cuts: int = 1, **kwargs):
        print(f"[MOCK] Mesh subdivivat cu {number_cuts} tăieturi")
        
    def separate(self, type: str = 'LOOSE'):
        print(f"[MOCK] Mesh separat: type='{type}'")


class MockObjectOps:
    """Simulează bpy.ops.object."""
    
    def __init__(self, context: MockContext):
        self._context = context
        
    def select_all(self, action: str = 'SELECT'):
        print(f"[MOCK] object.select_all(action='{action}')")
        
    def delete(self, **kwargs):
        print("[MOCK] Obiectele selectate au fost șterse")
        self._context._active_object = None
        
    def duplicate(self, linked: bool = False):
        if self._context._active_object:
            new_obj = MockObject(f"{self._context._active_object.name}.001", "MESH")
            self._context._active_object = new_obj
            print(f"[MOCK] Obiect duplicat: '{new_obj.name}'")
            
    def shade_smooth(self):
        print("[MOCK] Smooth shading aplicat")
        
    def mode_set(self, mode: str = 'OBJECT'):
        print(f"[MOCK] Mode setat: {mode}")
        
    def camera_add(self, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Camera", "CAMERA")
        obj.location = list(location)
        obj.data = MockCamera()
        self._context._active_object = obj
        print(f"[MOCK] Cameră creată la {location}")
        
    def light_add(self, type: str = 'POINT', location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject(f"Light_{type}", "LIGHT")
        obj.location = list(location)
        obj.data = MockLight()
        self._context._active_object = obj
        print(f"[MOCK] Lumină {type} creată la {location}")
        
    def text_add(self, location: Tuple = (0, 0, 0), **kwargs):
        obj = MockObject("Text", "FONT")
        obj.location = list(location)
        obj.data = type('MockTextData', (), {'body': '', 'size': 1, 'extrude': 0})()
        self._context._active_object = obj
        print(f"[MOCK] Text 3D creat la {location}")
        
    def join(self):
        print("[MOCK] Obiectele au fost unite")


class MockWMOps:
    """Simulează bpy.ops.wm."""
    
    def save_mainfile(self):
        print("[MOCK] Fișierul a fost salvat")
        
    def obj_export(self, filepath: str):
        print(f"[MOCK] Exportat ca OBJ: {filepath}")
        
    def stl_export(self, filepath: str):
        print(f"[MOCK] Exportat ca STL: {filepath}")


class MockRenderOps:
    """Simulează bpy.ops.render."""
    
    def render(self, write_still: bool = False):
        print("[MOCK] Render executat")


class MockExportOps:
    """Simulează bpy.ops.export_scene."""
    
    def fbx(self, filepath: str):
        print(f"[MOCK] Exportat ca FBX: {filepath}")


class MockPath:
    """Simulează bpy.path."""
    
    @staticmethod
    def abspath(path: str) -> str:
        return path.replace("//", "./")


# ==================== Instanțe globale ====================

context = MockContext()
data = MockData()
ops = MockOps(context)
path = MockPath()

print("[MOCK BPY] Modul mock bpy încărcat. Acesta simulează Blender API pentru testare.")
