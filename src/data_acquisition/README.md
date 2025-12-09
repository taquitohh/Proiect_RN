# 📁 Data Acquisition Module

## Descriere
Acest modul este responsabil pentru **achiziția și generarea datelor** de antrenare pentru sistemul Text-to-Blender.

## Structura
```
data_acquisition/
├── __init__.py
├── data_loader.py               # Funcții de încărcare și salvare date
├── generate_training_data.py    # Generator automat de date de antrenare
└── README.md                    # Acest fișier
```

## Funcționalități

### `data_loader.py`
- **`load_csv_data(filepath)`** - Încarcă date din fișiere CSV
- **`generate_synthetic_data(n_samples, n_features)`** - Generează date sintetice pentru testare
- **`get_data_info(df)`** - Returnează statistici despre dataset
- **`save_data(df, filepath)`** - Salvează date în format CSV

### `generate_training_data.py` ⭐ NOU
Generator automat de date de antrenare pentru clasificarea intențiilor:
- **1500+ samples** generate automat
- **109 intenții unice** (create_cube, apply_material, move_object, etc.)
- **16 categorii** de comenzi

#### Rulare generator:
```bash
cd e:\github\Proiect_RN
python src/data_acquisition/generate_training_data.py
```

#### Output:
- `data/generated/training_data.json` - Dataset în format JSON
- `data/generated/training_data.csv` - Dataset în format CSV

## Date Generate (100% Originale)

### Sursa datelor
Datele pentru antrenarea modelului Text-to-Blender sunt **100% originale**, create manual și generate automat:

1. **Scripturi Blender Python** (`data/raw/blender_scripts/`)
   - 40+ scripturi Python pentru operații Blender
   - Fiecare script reprezintă o acțiune specifică

2. **Dataset generat automat** (`data/generated/`)
   - 1500+ perechi text-intenție
   - Variații automate în limba română
   - 109 intenții unice

### Categorii de date generate:
| Categorie | Exemple | Samples |
|-----------|---------|---------|
| Creare obiecte | create_cube, create_sphere | ~225 |
| Materiale/Culori | apply_material_red, apply_material_metal | ~255 |
| Modificatori | add_modifier_bevel, add_modifier_mirror | ~120 |
| Transformări | move_object, rotate_object, scale_object | ~150 |
| Edit Mode | edit_extrude, edit_knife | ~90 |
| Duplicare | duplicate_object | ~75 |
| Render | render_scene | ~60 |
| Export | export_fbx, export_obj | ~60 |
| Help/Întrebări | help_general, help_commands | ~60 |

### Metodă de generare
```python
# Exemplu de sample generat
{
    "text": "creează un cub mare roșu",
    "intent": "create_cube",
    "params": {"size": "large", "color": "red"},
    "id": 1,
    "generated_at": "2025-12-09T..."
}
```

## Comenzi de rulare
```bash
# Generare date noi (1500 samples)
python src/data_acquisition/generate_training_data.py

# Încărcare date existente
python -c "from data_acquisition.data_loader import load_csv_data; print(load_csv_data('data/generated/training_data.csv'))"
```

## Relevanță pentru proiect
Datele sunt esențiale pentru:
1. **Clasificarea intenției** - Ce vrea utilizatorul să creeze
2. **Extragerea parametrilor** - Dimensiuni, culori, poziții
3. **Antrenarea RN** - Dataset pentru rețeaua neuronală
1. **Clasificarea intenției** - Ce vrea utilizatorul să creeze
2. **Extragerea parametrilor** - Dimensiuni, culori, poziții
3. **Generarea codului** - Script Python valid pentru Blender
