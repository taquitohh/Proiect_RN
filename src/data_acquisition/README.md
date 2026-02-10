# Data Acquisition

## Descriere
Generatorul produce dataset-uri sintetice, deterministe, pe baza unor intervale realiste
si reguli logice, cu seed fix pentru reproductibilitate. Pentru Etapa 3, focusul principal
este pe `chair`.

## Metoda de generare
- Esantionare determinista in intervale realiste pentru fiecare caracteristica geometrica.
- Aplicare reguli logice pentru combinatii valide (ex: spatar doar cand `has_backrest = 1`).
- Etichetare determinista in functie de parametri (mapping direct pe clase).

## Parametri folositi (chair)
- Intervale (exemple):
   - `seat_height`: 0.40–0.80
   - `seat_width`: 0.35–0.60
   - `seat_depth`: 0.35–0.60
   - `leg_count`: {3, 4, 5}
   - `leg_thickness`: 0.03–0.08
   - `has_backrest`: {0, 1}
   - `backrest_height`: 0.00 sau 0.20–0.50 (conditionat de `has_backrest`)
   - `style_variant`: {0, 1, 2}
- Reguli:
   - `backrest_height > 0` doar daca `has_backrest = 1`.
   - Valorile sunt generate strict in intervalele de ergonomie definite.
- Seed: fix (pentru reproductibilitate determinista).

## Justificare relevanta
Parametrii geometrici descriu direct variatiile reale ale unui scaun (dimensiuni, picioare,
spatar, stil). Dataset-ul sintetizat permite controlul distributiilor si etichetarea coerenta,
ceea ce sustine invatarea unei RN pentru clasificarea tipului de scaun pe baza geometriei.

## Scripturi disponibile
- `generate_chairs.py`
- `generate_tables.py`
- `generate_cabinets.py`
- `generate_fridges.py`
- `generate_stoves.py`

## Rulare (chair)
1. Activeaza mediul virtual.
2. Ruleaza:
   - `python src/data_acquisition/generate_chairs.py`

## Output
- CSV generat in: `data/generated/chairs_dataset.csv`

## Note
- Reguli: combinatii invalide sunt excluse (ex: backrest_height > 0 doar cand has_backrest = 1).
- Etichetele sunt atribuite determinist, pe baza parametrilor geometrici.
