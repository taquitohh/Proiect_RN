# 🧠 Proiect Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Etapa:** 3 - Analiza și Pregătirea Setului de Date  

---

## 📋 Descriere

Acest proiect implementează o platformă completă pentru antrenarea și evaluarea rețelelor neuronale, incluzând:

- **Backend Python** cu FastAPI pentru procesarea datelor și antrenarea modelelor
- **Frontend React/TypeScript** pentru vizualizare și interacțiune
- **Module de preprocesare** pentru curățarea și transformarea datelor
- **Implementare rețea neuronală** cu PyTorch

---

## 📁 Structura Proiectului

```
project/
├── README.md                 # Acest fișier
├── requirements.txt          # Dependențe Python
├── config/
│   ├── model_config.yaml     # Configurare model
│   └── preprocessing_config.yaml  # Configurare preprocesare
├── data/
│   ├── raw/                  # Date brute
│   ├── processed/            # Date curățate
│   ├── train/                # Set de instruire
│   ├── validation/           # Set de validare
│   ├── test/                 # Set de testare
│   └── README.md             # Descriere dataset
├── docs/
│   └── datasets/             # Documentație dataset
├── frontend/                 # Interfață web React
│   ├── src/
│   │   ├── api/              # Servicii API
│   │   ├── components/       # Componente React
│   │   ├── App.tsx           # Aplicația principală
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   └── vite.config.ts
├── models/                   # Modele salvate
└── src/
    ├── api.py                # API FastAPI
    ├── data_acquisition/     # Încărcare date
    ├── preprocessing/        # Preprocesare
    └── neural_network/       # Model RN
```

---

## 🚀 Instalare și Rulare

### Cerințe Preliminare

1. **Python 3.10+** - Descărcați de la https://www.python.org/downloads/
2. **Node.js 18+** - Descărcați de la https://nodejs.org/

### Pasul 1: Instalare Python

Descărcați și instalați Python de la https://www.python.org/downloads/

**Important:** La instalare, bifați opțiunea "Add Python to PATH"

### Pasul 2: Configurare mediu Python

```powershell
# Navigați la folderul proiectului
cd "e:\RN\Proiect"

# Creați un mediu virtual (opțional dar recomandat)
python -m venv venv

# Activați mediul virtual
.\venv\Scripts\Activate

# Instalați dependențele Python
pip install -r requirements.txt
```

### Pasul 3: Pornire Backend

```powershell
# Din folderul src
cd src

# Porniți serverul API
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Serverul va fi disponibil la: http://localhost:8000

### Pasul 4: Instalare și Pornire Frontend

```powershell
# Într-un terminal nou, navigați la frontend
cd "e:\RN\Proiect\frontend"

# Instalați dependențele Node.js
npm install

# Porniți aplicația în modul development
npm run dev
```

Aplicația va fi disponibilă la: http://localhost:3000

---

## 📖 Utilizare

### 1. Încărcare Date

- **Upload fișier CSV**: Încărcați propriul set de date
- **Generare date sintetice**: Creați date de test pentru experimentare

### 2. Preprocesare

- Configurați coloana țintă (target)
- Setați proporțiile train/validation/test
- Activați normalizarea și tratarea valorilor lipsă

### 3. Antrenare

- Configurați arhitectura (straturi ascunse)
- Setați hiperparametrii (epochs, batch size, learning rate)
- Urmăriți progresul antrenării

### 4. Rezultate

- Vizualizați graficele de loss și acuratețe
- Analizați performanța modelului

---

## 🔧 API Endpoints

| Endpoint | Metodă | Descriere |
|----------|--------|-----------|
| `/api/status` | GET | Starea curentă |
| `/api/data/upload` | POST | Încărcare fișier |
| `/api/data/generate` | POST | Generare date sintetice |
| `/api/data/info` | GET | Informații despre date |
| `/api/preprocess` | POST | Preprocesare date |
| `/api/train` | POST | Antrenare model |
| `/api/train/evaluate` | GET | Evaluare pe test |
| `/api/predict` | POST | Predicții |

---

## 📊 Configurare

### config/model_config.yaml

```yaml
model:
  architecture:
    hidden_layers: [128, 64, 32]
    activation: "relu"
    dropout: 0.2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### config/preprocessing_config.yaml

```yaml
splitting:
  train_ratio: 0.8
  validation_ratio: 0.1
  test_ratio: 0.1

preprocessing:
  normalization:
    enabled: true
    method: "minmax"
```

---

## 🛠️ Tehnologii Utilizate

### Backend
- **Python 3.10+**
- **PyTorch** - Framework deep learning
- **FastAPI** - API REST
- **pandas** - Manipulare date
- **scikit-learn** - Preprocesare

### Frontend
- **React 18** - UI Framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Recharts** - Grafice

---

## 📝 Etape Proiect

- [x] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Model antrenat și evaluat
- [ ] Documentație completă

---

## 📚 Referințe

- [PyTorch Documentation](https://pytorch.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

## ✉️ Contact

Student: [Nume Prenume]  
Email: [email@student.upb.ro]
