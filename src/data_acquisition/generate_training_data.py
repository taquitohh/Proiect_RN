"""
Generator automat de date de antrenare pentru Text-to-Blender.
==============================================================

Acest script generează automat perechi (text, intenție, parametri)
pentru antrenarea rețelei neuronale de clasificare intenții.

Rulare:
    python src/data_acquisition/generate_training_data.py

Output:
    - data/generated/training_data.json
    - data/generated/training_data.csv
"""

import json
import csv
import random
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Configurare paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "generated"


# ==================== TEMPLATE-URI PENTRU GENERARE ====================

# Verbe pentru acțiuni
VERBE_CREARE = ["creează", "fă", "adaugă", "generează", "pune", "desenează", "construiește", "vreau", "aș vrea", "bagă"]
VERBE_STERGERE = ["șterge", "elimină", "remove", "înlătură", "dă delete", "scapă de"]
VERBE_MODIFICARE = ["modifică", "schimbă", "transformă", "ajustează", "setează", "actualizează"]
VERBE_MUTARE = ["mută", "deplasează", "translatează", "poziționează", "plasează", "pune"]
VERBE_ROTIRE = ["rotește", "întoarce", "învârte", "rotiră"]
VERBE_SCALARE = ["scalează", "mărește", "micșorează", "redimensionează", "fă mai mare", "fă mai mic"]
VERBE_DUPLICARE = ["duplică", "copiază", "clonează", "multiplică", "fă o copie"]
VERBE_RENDER = ["renderează", "randează", "fă render", "generează imagine"]
VERBE_SAVE = ["salvează", "save", "păstrează", "exportă"]

# Obiecte 3D
OBIECTE = {
    "cube": ["cub", "cubul", "un cub", "cuburi", "cutie", "box"],
    "sphere": ["sferă", "sfera", "o sferă", "bilă", "glob", "minge"],
    "cylinder": ["cilindru", "cilindrul", "un cilindru", "tub", "țeavă"],
    "cone": ["con", "conul", "un con", "piramidă rotundă"],
    "torus": ["tor", "torus", "gogoașă", "inel", "cerc gros"],
    "plane": ["plan", "planul", "un plan", "suprafață", "podea", "floor"],
    "monkey": ["maimuță", "maimuța", "suzanne", "monkey", "cap de maimuță"],
    "icosphere": ["icosferă", "ico sferă", "sferă geometrică"],
    "grid": ["grilă", "grid", "rețea", "mesh"],
    "circle": ["cerc", "cercul", "un cerc", "inel subțire"],
    "text": ["text", "textul", "scrie", "literă", "cuvânt"],
    "camera": ["cameră", "camera", "aparat foto", "punct de vedere"],
    "light": ["lumină", "lumina", "lampă", "bec", "sursă de lumină"],
    "pyramid": ["piramidă", "piramida", "tetraedru"],
    "uv_sphere": ["sferă UV", "uv sphere", "sferă standard"],
    "bezier_curve": ["curbă", "curbă bezier", "linie curbată"],
    "nurbs_curve": ["curbă nurbs", "nurbs"],
    "empty": ["empty", "gol", "punct de referință", "locator"],
    "armature": ["armătură", "schelet", "bones", "oase"],
    "lattice": ["lattice", "rețea de deformare"]
}

# Culori
CULORI = {
    "red": ["roșu", "roșie", "roșii", "carmziu", "rubiniu"],
    "blue": ["albastru", "albastră", "albastre", "azuriu", "ceruleu"],
    "green": ["verde", "verzi", "smarald", "verdeață"],
    "yellow": ["galben", "galbenă", "galbene", "auriu deschis"],
    "orange": ["portocaliu", "portocalie", "oranj"],
    "purple": ["mov", "violet", "purpuriu", "lila"],
    "white": ["alb", "albă", "albe", "imaculat"],
    "black": ["negru", "neagră", "negre", "întunecat"],
    "pink": ["roz", "roză", "trandafiriu"],
    "brown": ["maro", "maroniu", "cafeniu", "ciocolatiu"],
    "gray": ["gri", "cenușiu", "argintiu închis"],
    "gold": ["auriu", "aurie", "de aur", "golden"],
    "silver": ["argintiu", "argintie", "de argint", "metalic deschis"],
    "cyan": ["cyan", "turcoaz", "albastru deschis"],
    "magenta": ["magenta", "roz închis", "fucsia"],
    "lime": ["lime", "verde deschis", "verde neon"],
    "navy": ["bleumarin", "albastru închis", "navy"],
    "olive": ["măsliniu", "olive", "verde închis"],
    "coral": ["coral", "somon", "piersică"],
    "beige": ["bej", "crem", "nisipiu"]
}

# Materiale
MATERIALE = {
    "metal": ["metalic", "metal", "de metal", "oțel", "fier"],
    "glass": ["sticlă", "de sticlă", "transparent", "cristal"],
    "wood": ["lemn", "de lemn", "lemnos", "parchet"],
    "plastic": ["plastic", "de plastic", "sintetic"],
    "rubber": ["cauciuc", "de cauciuc", "elastic"],
    "emission": ["strălucitor", "luminos", "emisiv", "neon", "glow"],
    "mirror": ["oglindă", "reflectiv", "reflectant"],
    "marble": ["marmură", "de marmură"],
    "concrete": ["beton", "de beton", "ciment"],
    "fabric": ["țesătură", "material textil", "pânză"],
    "leather": ["piele", "de piele"],
    "ceramic": ["ceramică", "de ceramică", "porțelan"],
    "chrome": ["crom", "cromat"],
    "copper": ["cupru", "de cupru", "arămiu"],
    "bronze": ["bronz", "de bronz"]
}

# Modificatori
MODIFICATORI = {
    "bevel": ["bevel", "rotunjit", "cu margini rotunjite", "șanfren"],
    "mirror": ["oglindă", "mirror", "simetric", "oglindit"],
    "array": ["array", "multiplicat", "repetat", "în serie", "multiplu"],
    "subsurf": ["subsurf", "subdivision", "neted", "smooth", "subdiviziune"],
    "solidify": ["solidify", "grosime", "solid", "îngroșat"],
    "boolean": ["boolean", "intersecție", "diferență", "uniune"],
    "decimate": ["decimate", "simplificat", "redus", "mai puține poligoane"],
    "wireframe": ["wireframe", "schelet", "sârmă"],
    "skin": ["skin", "piele", "înveliș"],
    "screw": ["screw", "spirală", "șurub"],
    "remesh": ["remesh", "re-topologie"],
    "displace": ["displace", "deplasare", "bump"],
    "wave": ["wave", "undă", "val"],
    "cloth": ["cloth", "simulare țesătură", "pânză"],
    "ocean": ["ocean", "apă", "mare"]
}

# Dimensiuni
DIMENSIUNI = {
    "small": ["mic", "mică", "mici", "micuț", "miniatură"],
    "medium": ["mediu", "medie", "normal", "standard"],
    "large": ["mare", "mari", "imens", "gigant", "enorm"],
    "tiny": ["minuscul", "foarte mic", "pitit", "microscopic"],
    "huge": ["uriaș", "enorm", "foarte mare", "colosal", "masiv"],
    "thin": ["subțire", "îngust", "slab"],
    "thick": ["gros", "lat", "robust"],
    "tall": ["înalt", "lung", "vertical"],
    "short": ["scurt", "scund", "mic în înălțime"],
    "wide": ["lat", "larg", "extins"]
}

# Poziții
POZITII = {
    "center": ["centru", "mijloc", "origine", "în centru"],
    "left": ["stânga", "în stânga", "pe stânga"],
    "right": ["dreapta", "în dreapta", "pe dreapta"],
    "up": ["sus", "deasupra", "în sus", "peste"],
    "down": ["jos", "dedesubt", "în jos", "sub"],
    "front": ["față", "în față", "înainte"],
    "back": ["spate", "în spate", "înapoi"],
    "top": ["vârf", "capăt", "sus de tot"],
    "bottom": ["bază", "fund", "jos de tot"],
    "corner": ["colț", "în colț"],
    "edge": ["margine", "pe margine"]
}

# Axe
AXE = {
    "x": ["x", "axa x", "orizontal", "pe x"],
    "y": ["y", "axa y", "adâncime", "pe y"],
    "z": ["z", "axa z", "vertical", "pe z", "înălțime"]
}

# Numere
NUMERE = {
    1: ["unu", "un", "o", "1", "unul"],
    2: ["doi", "două", "2", "doua"],
    3: ["trei", "3"],
    4: ["patru", "4"],
    5: ["cinci", "5"],
    6: ["șase", "6"],
    7: ["șapte", "7"],
    8: ["opt", "8"],
    9: ["nouă", "9"],
    10: ["zece", "10"],
    20: ["douăzeci", "20"],
    50: ["cincizeci", "50"],
    100: ["o sută", "100"]
}

# Acțiuni pentru scenă
SCENE_ACTIONS = {
    "new_scene": ["scenă nouă", "new scene", "resetează scena"],
    "render": ["render", "renderizează", "fă randare"],
    "save": ["salvează", "save", "păstrează proiectul"],
    "undo": ["undo", "anulează", "înapoi"],
    "redo": ["redo", "refă", "înainte"]
}

# Operații de editare
EDIT_OPERATIONS = {
    "extrude": ["extrudează", "extrude", "extinde", "scoate în afară"],
    "inset": ["inset", "inserează față", "înfundă"],
    "loop_cut": ["loop cut", "taie în cerc", "adaugă edge loop"],
    "knife": ["knife", "cuțit", "taie"],
    "bridge": ["bridge", "punte", "conectează"],
    "fill": ["fill", "umple", "închide"],
    "merge": ["merge", "unește", "combină vertices"],
    "separate": ["separate", "separă", "desparte"],
    "join": ["join", "alătură", "unește obiecte"]
}


# ==================== FUNCȚII DE GENERARE ====================

def generate_create_object_samples(n: int = 100) -> List[Dict]:
    """Generează samples pentru crearea de obiecte."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_CREARE)
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        
        # Variații de text
        templates = [
            f"{verb} {obj_name}",
            f"{verb}-mi {obj_name}",
            f"vreau {obj_name}",
            f"aș vrea {obj_name}",
            f"pot să am {obj_name}?",
            f"adaugă {obj_name} în scenă",
            f"pune {obj_name}",
            obj_name, # Adăugat: doar numele obiectului (ex: "cub", "sferă")
        ]
        
        # Opțional: adaugă dimensiune
        if random.random() > 0.5:
            dim_key = random.choice(list(DIMENSIUNI.keys()))
            dim_name = random.choice(DIMENSIUNI[dim_key])
            templates.extend([
                f"{verb} {obj_name} {dim_name}",
                f"{verb} un {obj_name} {dim_name}",
            ])
        
        text = random.choice(templates)
        
        samples.append({
            "text": text,
            "intent": f"create_{obj_key}",
            "params": {
                "object_type": obj_key,
                "size": random.choice([1.0, 2.0, 3.0, 5.0])
            }
        })
    
    return samples


def generate_apply_material_samples(n: int = 80) -> List[Dict]:
    """Generează samples pentru aplicarea de materiale/culori."""
    samples = []
    
    for _ in range(n):
        color_key = random.choice(list(CULORI.keys()))
        color_name = random.choice(CULORI[color_key])
        
        # Opțional: cu obiect specificat
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        
        templates = [
            f"colorează {color_name}",
            f"fă-l {color_name}",
            f"pune culoarea {color_name}",
            f"aplică material {color_name}",
            f"schimbă culoarea în {color_name}",
            f"vreau să fie {color_name}",
            f"{obj_name} {color_name}",
            f"creează {obj_name} {color_name}",
            f"fă {obj_name} {color_name}",
        ]
        
        text = random.choice(templates)
        
        samples.append({
            "text": text,
            "intent": f"apply_material_{color_key}",
            "params": {
                "color": color_key,
                "r": {"red": 1, "green": 0, "blue": 0, "yellow": 1, "white": 1, "black": 0}.get(color_key, 0.5),
                "g": {"red": 0, "green": 1, "blue": 0, "yellow": 1, "white": 1, "black": 0}.get(color_key, 0.5),
                "b": {"red": 0, "green": 0, "blue": 1, "yellow": 0, "white": 1, "black": 0}.get(color_key, 0.5),
            }
        })
    
    return samples


def generate_add_modifier_samples(n: int = 60) -> List[Dict]:
    """Generează samples pentru adăugarea de modificatori."""
    samples = []
    
    for _ in range(n):
        mod_key = random.choice(list(MODIFICATORI.keys()))
        mod_name = random.choice(MODIFICATORI[mod_key])
        
        templates = [
            f"adaugă modifier {mod_name}",
            f"aplică {mod_name}",
            f"fă-l {mod_name}",
            f"vreau să fie {mod_name}",
            f"pune {mod_name}",
            f"adaugă efect {mod_name}",
        ]
        
        text = random.choice(templates)
        
        samples.append({
            "text": text,
            "intent": f"add_modifier_{mod_key}",
            "params": {
                "modifier_type": mod_key
            }
        })
    
    return samples


def generate_transform_samples(n: int = 80) -> List[Dict]:
    """Generează samples pentru transformări (move, rotate, scale)."""
    samples = []
    
    # Move
    for _ in range(n // 3):
        verb = random.choice(VERBE_MUTARE)
        pos_key = random.choice(list(POZITII.keys()))
        pos_name = random.choice(POZITII[pos_key])
        
        templates = [
            f"{verb} {pos_name}",
            f"{verb}-l {pos_name}",
            f"deplasează spre {pos_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "move_object",
            "params": {
                "direction": pos_key,
                "distance": random.choice([1, 2, 3, 5])
            }
        })
    
    # Rotate
    for _ in range(n // 3):
        verb = random.choice(VERBE_ROTIRE)
        angle = random.choice([45, 90, 180, 270])
        
        templates = [
            f"{verb} cu {angle} grade",
            f"{verb}-l {angle}°",
            f"rotație {angle}",
            f"{verb}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "rotate_object",
            "params": {
                "angle": angle,
                "axis": random.choice(["x", "y", "z"])
            }
        })
    
    # Scale
    for _ in range(n // 3):
        verb = random.choice(VERBE_SCALARE)
        factor = random.choice([0.5, 1.5, 2, 3])
        
        templates = [
            f"{verb} de {factor} ori",
            f"{verb}-l",
            f"fă-l mai mare",
            f"fă-l mai mic",
            f"mărește",
            f"micșorează",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "scale_object",
            "params": {
                "factor": factor
            }
        })
    
    return samples


def generate_delete_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru ștergere."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_STERGERE)
        
        # Opțional: obiect specific
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTES[obj_key]) if random.random() > 0.5 else ""
        
        templates = [
            f"{verb} tot",
            f"{verb} totul",
            f"{verb} obiectul",
            f"{verb} selecția",
            f"curăță scena",
            f"golește scena",
        ]
        
        if obj_name:
            templates.append(f"{verb} {obj_name}")
        
        samples.append({
            "text": random.choice(templates),
            "intent": "delete_object",
            "params": {}
        })
    
    return samples


def generate_export_samples(n: int = 30) -> List[Dict]:
    """Generează samples pentru export."""
    samples = []
    
    formats = {
        "fbx": ["fbx", "FBX"],
        "obj": ["obj", "OBJ", "wavefront"],
        "stl": ["stl", "STL", "pentru printare 3D"]
    }
    
    for _ in range(n):
        fmt_key = random.choice(list(formats.keys()))
        fmt_name = random.choice(formats[fmt_key])
        
        templates = [
            f"exportă în {fmt_name}",
            f"salvează ca {fmt_name}",
            f"export {fmt_name}",
            f"fă export {fmt_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"export_{fmt_key}",
            "params": {
                "format": fmt_key
            }
        })
    
    return samples


def generate_combined_samples(n: int = 100) -> List[Dict]:
    """Generează samples cu comenzi combinate (obiect + culoare)."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_CREARE)
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        color_key = random.choice(list(CULORI.keys()))
        color_name = random.choice(CULORI[color_key])
        
        # Opțional: dimensiune
        dim_key = random.choice(list(DIMENSIUNI.keys()))
        dim_name = random.choice(DIMENSIUNI[dim_key])
        
        templates = [
            f"{verb} {obj_name} {color_name}",
            f"{verb} un {obj_name} {color_name}",
            f"{verb} {obj_name} {dim_name} {color_name}",
            f"vreau {obj_name} {color_name}",
            f"{obj_name} {color_name}",
            f"{obj_name} {dim_name} și {color_name}",
        ]
        
        text = random.choice(templates)
        
        samples.append({
            "text": text,
            "intent": f"create_{obj_key}",  # Intent principal
            "secondary_intent": f"apply_material_{color_key}",
            "params": {
                "object_type": obj_key,
                "color": color_key
            }
        })
    
    return samples


def generate_light_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru lumini."""
    samples = []
    
    light_types = {
        "point": ["punct", "point", "bec"],
        "sun": ["soare", "sun", "solar"],
        "spot": ["spot", "reflector", "spotlight"],
        "area": ["zonă", "area", "suprafață"]
    }
    
    for _ in range(n):
        light_key = random.choice(list(light_types.keys()))
        light_name = random.choice(light_types[light_key])
        verb = random.choice(VERBE_CREARE)
        
        templates = [
            f"{verb} lumină {light_name}",
            f"adaugă lumină de tip {light_name}",
            f"pune o lumină {light_name}",
            f"lumină {light_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"create_light_{light_key}",
            "params": {
                "light_type": light_key
            }
        })
    
    return samples


def generate_duplicate_samples(n: int = 50) -> List[Dict]:
    """Generează samples pentru duplicare obiecte."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_DUPLICARE)
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        num_key = random.choice([2, 3, 5])
        num_name = random.choice(NUMERE.get(num_key, [str(num_key)]))
        
        templates = [
            f"{verb} obiectul",
            f"{verb}-l",
            f"fă o copie",
            f"{verb} {obj_name}",
            f"fă {num_name} copii",
            f"{verb} de {num_name} ori",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "duplicate_object",
            "params": {
                "count": num_key
            }
        })
    
    return samples


def generate_render_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru render."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_RENDER)
        
        templates = [
            f"{verb}",
            f"{verb} scena",
            f"fă un render",
            f"generează imagine",
            f"vreau să văd rezultatul",
            f"arată-mi cum arată",
            f"fă o poză la scenă",
            f"render final",
            f"preview render",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "render_scene",
            "params": {}
        })
    
    return samples


def generate_edit_mode_samples(n: int = 60) -> List[Dict]:
    """Generează samples pentru operații în edit mode."""
    samples = []
    
    for _ in range(n):
        op_key = random.choice(list(EDIT_OPERATIONS.keys()))
        op_name = random.choice(EDIT_OPERATIONS[op_key])
        
        templates = [
            f"{op_name}",
            f"fă {op_name}",
            f"aplică {op_name}",
            f"vreau să {op_name}",
            f"folosește {op_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"edit_{op_key}",
            "params": {
                "operation": op_key
            }
        })
    
    return samples


def generate_material_type_samples(n: int = 60) -> List[Dict]:
    """Generează samples pentru tipuri de materiale."""
    samples = []
    
    for _ in range(n):
        mat_key = random.choice(list(MATERIALE.keys()))
        mat_name = random.choice(MATERIALE[mat_key])
        
        templates = [
            f"fă-l {mat_name}",
            f"aplică material {mat_name}",
            f"vreau să fie {mat_name}",
            f"pune textură de {mat_name}",
            f"schimbă în {mat_name}",
            f"material tip {mat_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"apply_material_{mat_key}",
            "params": {
                "material_type": mat_key
            }
        })
    
    return samples


def generate_scene_action_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru acțiuni pe scenă."""
    samples = []
    
    for _ in range(n):
        action_key = random.choice(list(SCENE_ACTIONS.keys()))
        action_name = random.choice(SCENE_ACTIONS[action_key])
        
        templates = [
            f"{action_name}",
            f"fă {action_name}",
            f"vreau {action_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"scene_{action_key}",
            "params": {}
        })
    
    return samples


def generate_complex_combined_samples(n: int = 80) -> List[Dict]:
    """Generează samples complexe cu multiple atribute."""
    samples = []
    
    for _ in range(n):
        verb = random.choice(VERBE_CREARE)
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        color_key = random.choice(list(CULORI.keys()))
        color_name = random.choice(CULORI[color_key])
        dim_key = random.choice(list(DIMENSIUNI.keys()))
        dim_name = random.choice(DIMENSIUNI[dim_key])
        mat_key = random.choice(list(MATERIALE.keys()))
        mat_name = random.choice(MATERIALE[mat_key])
        pos_key = random.choice(list(POZITII.keys()))
        pos_name = random.choice(POZITII[pos_key])
        
        templates = [
            f"{verb} {obj_name} {dim_name} {color_name}",
            f"{verb} {obj_name} {color_name} {mat_name}",
            f"{verb} {obj_name} {dim_name} în {pos_name}",
            f"vreau {obj_name} {dim_name} {color_name} {mat_name}",
            f"{obj_name} {dim_name} {color_name} poziționat {pos_name}",
            f"adaugă {obj_name} {mat_name} {color_name}",
            f"fă-mi {obj_name} {dim_name} {mat_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": f"create_{obj_key}",
            "params": {
                "object_type": obj_key,
                "color": color_key,
                "size": dim_key,
                "material": mat_key,
                "position": pos_key
            }
        })
    
    return samples


def generate_question_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru întrebări/help."""
    samples = []
    
    questions = [
        ("cum fac un cub", "help_create_cube"),
        ("cum creez o sferă", "help_create_sphere"),
        ("cum aplic un material", "help_apply_material"),
        ("cum șterg un obiect", "help_delete"),
        ("cum rotesc", "help_rotate"),
        ("cum scalez", "help_scale"),
        ("cum export în fbx", "help_export"),
        ("cum adaug lumină", "help_light"),
        ("ce pot face", "help_general"),
        ("ajutor", "help_general"),
        ("help", "help_general"),
        ("cum funcționează", "help_general"),
        ("ce comenzi ai", "help_commands"),
        ("arată-mi comenzile", "help_commands"),
        ("ce știi să faci", "help_capabilities"),
    ]
    
    for _ in range(n):
        text, intent = random.choice(questions)
        
        # Variații
        prefixes = ["", "hei, ", "salut, ", "te rog, "]
        suffixes = ["", "?", " te rog", " vă rog"]
        
        final_text = random.choice(prefixes) + text + random.choice(suffixes)
        
        samples.append({
            "text": final_text,
            "intent": intent,
            "params": {}
        })
    
    return samples


def generate_select_samples(n: int = 40) -> List[Dict]:
    """Generează samples pentru selecție."""
    samples = []
    
    for _ in range(n):
        obj_key = random.choice(list(OBIECTE.keys()))
        obj_name = random.choice(OBIECTE[obj_key])
        
        templates = [
            f"selectează {obj_name}",
            f"alege {obj_name}",
            f"click pe {obj_name}",
            "selectează tot",
            "selectează totul",
            "deselectează",
            "deselectează tot",
            f"selectează obiectul {obj_name}",
        ]
        
        samples.append({
            "text": random.choice(templates),
            "intent": "select_object",
            "params": {
                "object_type": obj_key if random.random() > 0.3 else "all"
            }
        })
    
    return samples


# ==================== FUNCȚIA PRINCIPALĂ ====================

def generate_dataset(total_samples: int = 500) -> List[Dict]:
    """
    Generează un dataset complet pentru antrenare.
    
    Args:
        total_samples: Numărul total de samples de generat
        
    Returns:
        Lista cu toate samples generate
    """
    all_samples = []
    
    # Distribuție pe categorii (trebuie să fie 1.0 în total)
    distribution = {
        "create_object": 0.15,
        "apply_material": 0.12,
        "add_modifier": 0.08,
        "transform": 0.10,
        "delete": 0.04,
        "export": 0.04,
        "combined": 0.10,
        "light": 0.04,
        "duplicate": 0.05,
        "render": 0.04,
        "edit_mode": 0.06,
        "material_type": 0.05,
        "scene_action": 0.04,
        "complex_combined": 0.05,
        "question": 0.04,
        "select": 0.04
    }
    
    all_samples.extend(generate_create_object_samples(int(total_samples * distribution["create_object"])))
    all_samples.extend(generate_apply_material_samples(int(total_samples * distribution["apply_material"])))
    all_samples.extend(generate_add_modifier_samples(int(total_samples * distribution["add_modifier"])))
    all_samples.extend(generate_transform_samples(int(total_samples * distribution["transform"])))
    all_samples.extend(generate_delete_samples(int(total_samples * distribution["delete"])))
    all_samples.extend(generate_export_samples(int(total_samples * distribution["export"])))
    all_samples.extend(generate_combined_samples(int(total_samples * distribution["combined"])))
    all_samples.extend(generate_light_samples(int(total_samples * distribution["light"])))
    all_samples.extend(generate_duplicate_samples(int(total_samples * distribution["duplicate"])))
    all_samples.extend(generate_render_samples(int(total_samples * distribution["render"])))
    all_samples.extend(generate_edit_mode_samples(int(total_samples * distribution["edit_mode"])))
    all_samples.extend(generate_material_type_samples(int(total_samples * distribution["material_type"])))
    all_samples.extend(generate_scene_action_samples(int(total_samples * distribution["scene_action"])))
    all_samples.extend(generate_complex_combined_samples(int(total_samples * distribution["complex_combined"])))
    all_samples.extend(generate_question_samples(int(total_samples * distribution["question"])))
    all_samples.extend(generate_select_samples(int(total_samples * distribution["select"])))
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Adaugă metadata
    for i, sample in enumerate(all_samples):
        sample["id"] = i + 1
        sample["generated_at"] = datetime.now().isoformat()
    
    return all_samples


def save_dataset(samples: List[Dict], output_dir: Path = DATA_DIR):
    """Salvează dataset-ul în JSON și CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_path = output_dir / "training_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "total_samples": len(samples),
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "samples": samples
        }, f, ensure_ascii=False, indent=2)
    print(f"✅ Salvat: {json_path} ({len(samples)} samples)")
    
    # CSV pentru analiză
    csv_path = output_dir / "training_data.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "intent", "params"])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "id": sample["id"],
                "text": sample["text"],
                "intent": sample["intent"],
                "params": json.dumps(sample.get("params", {}), ensure_ascii=False)
            })
    print(f"✅ Salvat: {csv_path}")
    
    # Statistici
    print_statistics(samples)


def print_statistics(samples: List[Dict]):
    """Afișează statistici despre dataset."""
    print("\n" + "="*50)
    print("📊 STATISTICI DATASET GENERAT")
    print("="*50)
    
    # Count per intent
    intent_counts = {}
    for sample in samples:
        intent = sample["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print(f"\n📌 Total samples: {len(samples)}")
    print(f"\n📌 Distribuție per intenție:")
    
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        pct = count / len(samples) * 100
        bar = "█" * int(pct / 2)
        print(f"  {intent:30s} {count:4d} ({pct:5.1f}%) {bar}")
    
    print(f"\n📌 Număr unic de intenții: {len(intent_counts)}")


# Fix typo in generate_delete_samples
OBIECTES = OBIECTE  # Alias pentru typo


if __name__ == "__main__":
    print("🚀 Generare date de antrenare pentru Text-to-Blender...")
    print("-" * 50)
    
    # Generează 1500 samples (mai multe pentru antrenare mai bună)
    samples = generate_dataset(total_samples=1500)
    
    # Salvează
    save_dataset(samples)
    
    print("\n✅ Generare completă!")
    print(f"📁 Fișierele sunt în: {DATA_DIR}")
