# 📊 Descrierea Setului de Date

## Sursa datelor

* **Origine:** [Descrieți sursa datelor - ex: senzori robot, dataset public, simulare]
* **Modul de achiziție:** [Senzori reali / Simulare / Fișier extern / Generare programatică]
* **Perioada / condițiile colectării:** [Ex: Noiembrie 2024 - Ianuarie 2025]

## Caracteristicile dataset-ului

* **Număr total de observații:** [Ex: 15,000]
* **Număr de caracteristici (features):** [Ex: 12]
* **Tipuri de date:** [Numerice / Categoriale / Temporale / Imagini]
* **Format fișiere:** [CSV / TXT / JSON / PNG / Altele]

## Descrierea caracteristicilor

| **Caracteristică** | **Tip** | **Unitate** | **Descriere** | **Domeniu valori** |
|-------------------|---------|-------------|---------------|--------------------|
| feature_1 | numeric | mm | [...] | 0–150 |
| feature_2 | categorial | – | [...] | {A, B, C} |
| feature_3 | numeric | m/s | [...] | 0–2.5 |

## Structura folderelor

```
data/
├── raw/               # Date brute, nemodificate
├── processed/         # Date curățate și transformate
├── train/             # Set de instruire (70-80%)
├── validation/        # Set de validare (10-15%)
└── test/              # Set de testare (10-15%)
```

## Procesarea datelor

### Împărțirea seturilor
- **Train:** 80% din date
- **Validation:** 10% din date
- **Test:** 10% din date

### Transformări aplicate
- [ ] Normalizare Min-Max / Standardizare
- [ ] Encoding pentru variabile categoriale
- [ ] Tratarea valorilor lipsă
- [ ] Eliminarea outlierilor
