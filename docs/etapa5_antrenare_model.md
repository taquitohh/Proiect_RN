# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** [Nume Prenume]  
**Link Repository GitHub:** https://github.com/taquitohh/Proiect_RN  
**Data:** Decembrie 2024

---

## 🎯 Rezultate Antrenare - REZUMAT

| **Metrică** | **Valoare** | **Target** | **Status** |
|-------------|-------------|------------|------------|
| **Accuracy** | 75.64% | ≥65% | ✅ **ATINS** |
| **F1 Score (macro)** | 0.6032 | ≥0.60 | ✅ **ATINS** |
| **F1 Score (weighted)** | 0.7311 | - | ✅ |
| **Top-3 Accuracy** | 82.91% | - | 🎉 Bonus |
| **Top-5 Accuracy** | 85.47% | - | 🎉 Bonus |

---

## 📊 Configurație Antrenare

### Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare** | **Justificare** |
|--------------------|-------------|-----------------|
| **Learning rate** | 0.001 | Valoare standard pentru Adam, asigură convergență stabilă pentru clasificare multi-class |
| **Batch size** | 32 | Cu 1092 train samples → 34 iterații/epocă. Compromis optim memorie/stabilitate gradient |
| **Epochs** | 100 (max) | Cu early stopping; a rulat efectiv 36 epoci |
| **Optimizer** | Adam | Adaptive learning rate, performant pentru rețele feed-forward cu 3 straturi hidden |
| **Loss function** | CrossEntropyLoss | Standard pentru clasificare multi-class cu 109 clase |
| **Hidden layers** | [128, 64, 32] | Piramidă descrescătoare pentru compresie progresivă features |
| **Activation** | ReLU | Evită problema vanishing gradient, rapid de calculat |
| **Dropout** | 0.2 | Regularizare pentru prevenire overfitting la dataset mic |
| **Early stopping patience** | 10 | Oprește antrenarea după 10 epoci fără îmbunătățire val_loss |

### Justificare Detaliată Batch Size

```
Am ales batch_size=32 pentru că avem N=1092 train samples → 1092/32 ≈ 34 iterații/epocă.
Aceasta oferă un echilibru între:
- Stabilitate gradient (batch prea mic → zgomot mare în gradient)
- Memorie CPU (nu avem GPU, deci memory constraints reduse)
- Timp antrenare (batch 32 asigură convergență în 36 epoci pentru 109 clase)
- Early stopping a oprit antrenarea înainte de overfitting
```

---

## 📈 Rezultate Detaliate

### Statistici Antrenare

| **Parametru** | **Valoare** |
|---------------|-------------|
| Epoci rulate | 36 (din 100 max) |
| Timp total antrenare | 3.17 secunde |
| Device | CPU |
| Train samples | 1,092 |
| Validation samples | 234 |
| Test samples | 234 |
| Număr clase | 109 intenții unice |
| Vocabular | 523 cuvinte unice |

### Evoluție Loss și Accuracy

| Epocă | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 10 | 2.6723 | 31.41% | 2.8616 | 37.18% |
| 20 | 1.3618 | 61.17% | 2.1903 | 62.82% |
| 30 | 0.8454 | 74.08% | 2.2014 | 69.66% |
| 36 (final) | 0.6781 | 79.21% | 2.2817 | 72.65% |

### Metrici Test Set Complete

```json
{
  "test_loss": 1.7355,
  "accuracy": 0.7564,
  "f1_macro": 0.6032,
  "f1_weighted": 0.7311,
  "precision_macro": 0.6190,
  "recall_macro": 0.6418,
  "top_3_accuracy": 0.8291,
  "top_5_accuracy": 0.8547,
  "num_test_samples": 234,
  "num_classes": 109
}
```

---

## 🔍 Analiză Erori în Context Industrial

### 1. Pe ce clase greșește cel mai mult modelul?

```
Analiză din error_analysis.csv arată 57 erori din 234 samples (24.36%).

Top 5 confuzii:
1. move_object → rotate_object: 2 erori
   Cauză: Ambele sunt operații de transformare cu sintaxă similară ("mută", "rotește")

2. add_modifier_cloth → apply_material_fabric: 1 eroare
   Cauză: Semantică similară ("cloth" vs "fabric")

3. add_modifier_ocean → edit_bridge: 1 eroare
   Cauză: Vocabular limitat pentru comenzi rare

4. add_modifier_mirror → apply_material_brown: 1 eroare
   Cauză: Clasificare incorectă pentru clase cu puține samples

5. add_modifier_screw → add_modifier_subsurf: 1 eroare
   Cauză: Ambele sunt modifiers, structură comandă similară
```

### 2. Ce caracteristici ale datelor cauzează erori?

```
Analiza arată că:
- Clase cu <5 samples în train au accuracy sub 50%
- Comenzile scurte (1-2 cuvinte) au mai multe erori
- Sinonime românești ("mișcă" vs "deplasează") confundă modelul
- Comenzile cu context lipsă ("fă un cub" vs "creează un cub basic") sunt ambigue
```

### 3. Ce implicații are pentru aplicația Text-to-Blender?

```
FALSE NEGATIVES (comandă nerecunoscută):
- Impact: Utilizatorul trebuie să reformuleze
- Severitate: MEDIE - utilizatorul poate reîncerca

FALSE POSITIVES (comandă incorect clasificată):
- Impact: Se generează cod Blender incorect
- Severitate: JOASĂ - utilizatorul poate vizualiza rezultatul și anula

Prioritate: Minimizare confuzii între comenzi destructive (delete_all) și 
comenzi constructive (create_*). Model-ul actual nu confundă aceste categorii.
```

### 4. Ce măsuri corective propunem?

```
Măsuri concrete pentru îmbunătățire:
1. AUGMENTARE DATE: Generare 50+ variante suplimentare pentru clasele minoritare
2. SINONIME: Extindere vocabular cu sinonime românești (mută/mișcă/deplasează)
3. N-GRAMS: Adăugare bigrams pentru context mai bun ("cub mare" vs "cub basic")
4. CLASS WEIGHTS: Aplicare weights inverse proporționale cu frecvența clasei
5. ENSEMBLE: Combinare cu model bazat pe reguli pentru comenzi simple
```

---

## 📁 Fișiere Rezultate Salvate

| **Fișier** | **Descriere** |
|------------|---------------|
| `models/trained_model.pt` | Model PyTorch antrenat (81,005 parametri) |
| `results/training_history.csv` | Istoric complet 36 epoci |
| `results/test_metrics.json` | Metrici evaluare test set |
| `results/evaluation_metrics.json` | Metrici detaliate cu top-k |
| `results/training_curves.png` | Grafic loss și accuracy |
| `results/confusion_matrix.png` | Matricea de confuzie (top 20 clase) |
| `results/per_class_metrics.csv` | Precision/recall per clasă |
| `results/error_analysis.csv` | Lista erorilor cu true/predicted |
| `results/test_class_distribution.png` | Distribuția claselor în test |

---

## 🚀 Instrucțiuni Rulare

### Antrenare Model

```bash
# Activare environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Antrenare
python -m src.neural_network.train
```

### Evaluare Model

```bash
python -m src.neural_network.evaluate
```

### Output Așteptat

```
============================================================
🧠 Text-to-Blender Neural Network Training
============================================================
✅ Date încărcate:
   Train: 1092 samples, input_size=523
   Val:   234 samples
   Test:  234 samples
...
🎯 Verificare obiective Etapa 5:
   Accuracy ≥ 65%: ✅ DA (75.64%)
   F1 ≥ 0.60:      ✅ DA (0.6032)
```

---

## ✅ Checklist Etapa 5

### Nivel 1 - OBLIGATORIU (70%)

- [x] Model antrenat de la ZERO (nu fine-tuning)
- [x] Minimum 10 epoci rulate (36 epoci efectiv)
- [x] Tabel hiperparametri + justificări completat
- [x] **Accuracy ≥65%** → 75.64% ✅
- [x] **F1 ≥0.60** → 0.6032 ✅
- [x] Model salvat în `models/trained_model.pt`
- [x] `results/training_history.csv` cu toate epocile

### Nivel 2 - RECOMANDAT (85-90%)

- [x] **Early Stopping** implementat (patience=10)
- [x] Grafic loss/accuracy în `results/training_curves.png`
- [x] Analiză erori context industrial (4 secțiuni completate)
- [x] **Accuracy ≥75%** → 75.64% ✅
- [ ] **F1 ≥0.70** → 0.6032 (nu atins, dar aproape)

### Nivel 3 - BONUS (100%)

- [x] Confusion Matrix + analiză erori → `results/confusion_matrix.png`
- [x] Top-k accuracy calculat (Top-3: 82.91%, Top-5: 85.47%)
- [x] Per-class metrics → `results/per_class_metrics.csv`
- [ ] Export ONNX/TFLite (TODO)
- [ ] Comparație 2+ arhitecturi (TODO)

---

## 📊 Grafice Antrenare

### Training Curves
![Training Curves](results/training_curves.png)

### Confusion Matrix (Top 20 clase)
![Confusion Matrix](results/confusion_matrix.png)

### Distribuție Clase Test
![Test Class Distribution](results/test_class_distribution.png)

---

## 🏷️ Versiune și Commit

```bash
# Commit final
git add .
git commit -m "Etapa 5 completă – Accuracy=75.64%, F1=0.6032"

# Tag versiune
git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat cu succes"

# Push
git push origin main --tags
```

---

**✅ Etapa 5 completată cu succes!**

Modelul Text-to-Blender atinge obiectivele minime și oferă o bază solidă pentru 
integrarea în aplicația completă (Etapa 6).
