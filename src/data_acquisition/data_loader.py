"""
Modul pentru achiziția și generarea datelor.
============================================

Acest modul conține funcții pentru:
- Încărcarea datelor din diverse surse
- Generarea datelor sintetice pentru testare
- Descărcarea dataset-urilor publice
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
import yaml
import json


def load_config(config_path: str = "../../config/preprocessing_config.yaml") -> Dict[str, Any]:
    """Încarcă configurația din fișierul YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_training_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Încarcă datele de antrenament din JSON sau CSV.
    Format așteptat: [{"text": "...", "intent": "..."}]
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
        return data
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        records: List[Dict[str, Any]] = df.to_dict(orient='records')  # type: ignore
        return records
    else:
        raise ValueError("Format fișier neacceptat. Folosiți .json sau .csv")

def load_csv_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Încarcă date din fișier CSV.
    
    Args:
        file_path: Calea către fișierul CSV
        **kwargs: Argumente adiționale pentru pandas.read_csv
        
    Returns:
        DataFrame cu datele încărcate
    """
    return pd.read_csv(file_path, **kwargs)


def load_json_data(file_path: str) -> pd.DataFrame:
    """
    Încarcă date din fișier JSON.
    
    Args:
        file_path: Calea către fișierul JSON
        
    Returns:
        DataFrame cu datele încărcate
    """
    return pd.read_json(file_path)


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    noise: float = 0.1,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generează date sintetice pentru testare.
    
    Args:
        n_samples: Numărul de eșantioane
        n_features: Numărul de caracteristici
        n_classes: Numărul de clase (pentru clasificare)
        noise: Nivelul de zgomot
        random_seed: Seed pentru reproducibilitate
        
    Returns:
        Tuple cu DataFrame de caracteristici și Series de etichete
    """
    np.random.seed(random_seed)
    
    # Generare caracteristici
    X = np.random.randn(n_samples, n_features)
    
    # Generare etichete pe baza unor combinații de caracteristici
    weights = np.random.randn(n_features)
    scores = X @ weights + noise * np.random.randn(n_samples)
    
    # Convertire la clase
    percentiles = np.percentile(scores, np.linspace(0, 100, n_classes + 1)[1:-1])
    y = np.digitize(scores, percentiles)
    
    # Creare DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    labels = pd.Series(y, name='target')
    
    return df, labels


def save_data(
    data: pd.DataFrame,
    output_path: str,
    format: str = 'csv'
) -> None:
    """
    Salvează datele în formatul specificat.
    
    Args:
        data: DataFrame de salvat
        output_path: Calea de salvare
        format: Formatul fișierului ('csv' sau 'json')
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'csv':
        data.to_csv(output_path, index=False)
    elif format == 'json':
        data.to_json(output_path, orient='records', indent=2)
    else:
        raise ValueError(f"Format necunoscut: {format}")
    
    print(f"Date salvate în: {output_path}")


def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Returnează informații despre dataset.
    
    Args:
        data: DataFrame de analizat
        
    Returns:
        Dicționar cu informații despre date
    """
    return {
        'n_samples': len(data),
        'n_features': len(data.columns),
        'columns': list(data.columns),
        'dtypes': data.dtypes.astype(str).to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2  # MB
    }


if __name__ == "__main__":
    # Exemplu de utilizare
    print("Generare date sintetice pentru testare...")
    X, y = generate_synthetic_data(n_samples=1000, n_features=10, n_classes=3)
    
    # Combinare caracteristici și etichete
    data = pd.concat([X, y], axis=1)
    
    # Afișare informații
    info = get_data_info(data)
    print(f"Număr eșantioane: {info['n_samples']}")
    print(f"Număr caracteristici: {info['n_features']}")
    print(f"Utilizare memorie: {info['memory_usage']:.2f} MB")
    
    # Salvare
    save_data(data, "../../data/raw/synthetic_data.csv")
