# UI Flask

## Descriere
Interfata web permite introducerea parametrilor geometrici, ruleaza inferenta RN si afiseaza
clasa prezisa, probabilitatile si scriptul Blender generat. Optional, cere un preview de randare
prin Blender API.

## Rulare
1. Activeaza mediul virtual.
2. Porneste Blender API (daca vrei preview):
   - `python src/blender_api/server.py`
3. Porneste UI:
   - `python src/app/main.py`
4. Deschide: http://127.0.0.1:5000

## Note
- Daca Blender API nu este pornit, UI ramane functional fara preview.
- Endpoint de status: `/status`.
