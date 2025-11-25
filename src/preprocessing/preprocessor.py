"""
Modul pentru preprocesarea datelor.
====================================

Acest modul conține funcții pentru:
- Curățarea datelor (valori lipsă, duplicate, outlieri)
- Transformarea caracteristicilor (normalizare, encoding)
- Împărțirea în seturi train/validation/test
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
import yaml
import joblib


class DataPreprocessor:
    """Clasă pentru preprocesarea datelor."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inițializare preprocessor.
        
        Args:
            config_path: Calea către fișierul de configurare
        """
        self.config = self._load_config(config_path)
        self.scalers = {}
        self.encoders = {}
        self.fitted = False
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Încarcă configurația."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Returnează configurația implicită."""
        return {
            'splitting': {
                'train_ratio': 0.8,
                'validation_ratio': 0.1,
                'test_ratio': 0.1,
                'random_seed': 42,
                'stratify': True
            },
            'preprocessing': {
                'normalization': {'enabled': True, 'method': 'minmax'},
                'missing_values': {'strategy': 'median', 'threshold': 0.3},
                'outliers': {'enabled': True, 'method': 'iqr', 'iqr_multiplier': 1.5}
            }
        }
    
    def analyze_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analizează datele și returnează statistici.
        
        Args:
            data: DataFrame de analizat
            
        Returns:
            Dicționar cu statistici și probleme identificate
        """
        analysis = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'missing_values': {},
            'duplicates': data.duplicated().sum(),
            'statistics': {},
            'issues': []
        }
        
        # Analiza valorilor lipsă
        for col in data.columns:
            missing = data[col].isnull().sum()
            missing_pct = missing / len(data) * 100
            analysis['missing_values'][col] = {
                'count': int(missing),
                'percentage': round(missing_pct, 2)
            }
            if missing_pct > 0:
                analysis['issues'].append(f"Coloana '{col}' are {missing_pct:.2f}% valori lipsă")
        
        # Statistici pentru coloane numerice
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            analysis['statistics'][col] = {
                'mean': float(data[col].mean()),
                'median': float(data[col].median()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'q25': float(data[col].quantile(0.25)),
                'q75': float(data[col].quantile(0.75))
            }
        
        return analysis
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = 'median',
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Tratează valorile lipsă.
        
        Args:
            data: DataFrame cu date
            strategy: Strategia de imputare ('mean', 'median', 'mode', 'drop')
            threshold: Pragul pentru eliminarea coloanelor
            
        Returns:
            DataFrame cu valorile lipsă tratate
        """
        data = data.copy()
        
        # Elimină coloanele cu prea multe valori lipsă
        missing_pct = data.isnull().sum() / len(data)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            print(f"Eliminare coloane cu > {threshold*100}% valori lipsă: {cols_to_drop}")
            data = data.drop(columns=cols_to_drop)
        
        # Imputare pentru restul coloanelor
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                if data[col].dtype in ['float64', 'int64']:
                    if strategy == 'mean':
                        data[col].fillna(data[col].mean(), inplace=True)
                    elif strategy == 'median':
                        data[col].fillna(data[col].median(), inplace=True)
                    elif strategy == 'drop':
                        data = data.dropna(subset=[col])
                else:
                    # Pentru coloane categoriale, folosim modul
                    data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'UNKNOWN', inplace=True)
        
        return data
    
    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Elimină rândurile duplicate."""
        n_duplicates = data.duplicated().sum()
        if n_duplicates > 0:
            print(f"Eliminare {n_duplicates} duplicate")
            data = data.drop_duplicates()
        return data
    
    def handle_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        iqr_multiplier: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Tratează valorile outlier.
        
        Args:
            data: DataFrame cu date
            method: Metoda de detectare ('iqr' sau 'percentile')
            iqr_multiplier: Multiplicator pentru metoda IQR
            columns: Coloanele de procesat (None = toate numerice)
            
        Returns:
            DataFrame cu outlieri tratați
        """
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_multiplier * IQR
                upper = Q3 + iqr_multiplier * IQR
            else:  # percentile
                lower = data[col].quantile(0.01)
                upper = data[col].quantile(0.99)
            
            # Limitare valori (capping)
            data[col] = data[col].clip(lower=lower, upper=upper)
        
        return data
    
    def normalize(
        self,
        data: pd.DataFrame,
        method: str = 'minmax',
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalizează datele numerice.
        
        Args:
            data: DataFrame cu date
            method: Metoda de normalizare ('minmax' sau 'standard')
            columns: Coloanele de normalizat
            fit: Dacă să antreneze scaler-ul (True pentru train, False pentru val/test)
            
        Returns:
            DataFrame cu date normalizate
        """
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            if method == 'minmax':
                self.scalers['main'] = MinMaxScaler()
            else:
                self.scalers['main'] = StandardScaler()
            data[columns] = self.scalers['main'].fit_transform(data[columns])
        else:
            if 'main' in self.scalers:
                data[columns] = self.scalers['main'].transform(data[columns])
        
        return data
    
    def encode_categorical(
        self,
        data: pd.DataFrame,
        method: str = 'onehot',
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Codifică variabilele categoriale.
        
        Args:
            data: DataFrame cu date
            method: Metoda de codificare ('onehot' sau 'label')
            columns: Coloanele de codificat
            fit: Dacă să antreneze encoder-ul
            
        Returns:
            DataFrame cu variabile codificate
        """
        data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in columns:
            if fit:
                if method == 'label':
                    self.encoders[col] = LabelEncoder()
                    data[col] = self.encoders[col].fit_transform(data[col].astype(str))
                else:
                    # One-hot encoding
                    dummies = pd.get_dummies(data[col], prefix=col)
                    data = pd.concat([data.drop(columns=[col]), dummies], axis=1)
            else:
                if col in self.encoders:
                    data[col] = self.encoders[col].transform(data[col].astype(str))
        
        return data
    
    def split_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        random_seed: int = 42
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Împarte datele în seturi train/validation/test.
        
        Args:
            data: DataFrame complet
            target_column: Numele coloanei țintă
            train_ratio: Proporția pentru antrenare
            validation_ratio: Proporția pentru validare
            test_ratio: Proporția pentru testare
            stratify: Dacă să stratifice (pentru clasificare)
            random_seed: Seed pentru reproducibilitate
            
        Returns:
            Dicționar cu seturile de date
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        stratify_param = y if stratify else None
        
        # Prima împărțire: train vs (validation + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(validation_ratio + test_ratio),
            random_state=random_seed,
            stratify=stratify_param
        )
        
        # A doua împărțire: validation vs test
        val_test_ratio = validation_ratio / (validation_ratio + test_ratio)
        stratify_temp = y_temp if stratify else None
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_test_ratio),
            random_state=random_seed,
            stratify=stratify_temp
        )
        
        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def preprocess_pipeline(
        self,
        data: pd.DataFrame,
        target_column: str,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Pipeline complet de preprocesare.
        
        Args:
            data: DataFrame original
            target_column: Coloana țintă
            numerical_columns: Coloane numerice
            categorical_columns: Coloane categoriale
            
        Returns:
            Dicționar cu seturile procesate și metadate
        """
        config = self.config
        
        print("=== Pipeline de Preprocesare ===")
        
        # 1. Analiza inițială
        print("\n1. Analiză date...")
        analysis = self.analyze_data(data)
        
        # 2. Curățare date
        print("\n2. Curățare date...")
        data = self.remove_duplicates(data)
        data = self.handle_missing_values(
            data,
            strategy=config['preprocessing']['missing_values']['strategy'],
            threshold=config['preprocessing']['missing_values']['threshold']
        )
        
        # 3. Tratare outlieri
        if config['preprocessing']['outliers']['enabled']:
            print("\n3. Tratare outlieri...")
            data = self.handle_outliers(
                data,
                method=config['preprocessing']['outliers']['method'],
                iqr_multiplier=config['preprocessing']['outliers'].get('iqr_multiplier', 1.5),
                columns=numerical_columns
            )
        
        # 4. Împărțire date
        print("\n4. Împărțire date în train/validation/test...")
        splits = self.split_data(
            data,
            target_column=target_column,
            train_ratio=config['splitting']['train_ratio'],
            validation_ratio=config['splitting']['validation_ratio'],
            test_ratio=config['splitting']['test_ratio'],
            stratify=config['splitting']['stratify'],
            random_seed=config['splitting']['random_seed']
        )
        
        # 5. Normalizare (fit pe train, transform pe val/test)
        if config['preprocessing']['normalization']['enabled']:
            print("\n5. Normalizare date...")
            X_train, y_train = splits['train']
            X_val, y_val = splits['validation']
            X_test, y_test = splits['test']
            
            X_train = self.normalize(X_train, method=config['preprocessing']['normalization']['method'], fit=True)
            X_val = self.normalize(X_val, method=config['preprocessing']['normalization']['method'], fit=False)
            X_test = self.normalize(X_test, method=config['preprocessing']['normalization']['method'], fit=False)
            
            splits = {
                'train': (X_train, y_train),
                'validation': (X_val, y_val),
                'test': (X_test, y_test)
            }
        
        self.fitted = True
        
        print("\n=== Preprocesare completă ===")
        print(f"Train: {len(splits['train'][0])} eșantioane")
        print(f"Validation: {len(splits['validation'][0])} eșantioane")
        print(f"Test: {len(splits['test'][0])} eșantioane")
        
        return {
            'splits': splits,
            'analysis': analysis,
            'config': config
        }
    
    def save_preprocessor(self, path: str):
        """Salvează starea preprocessor-ului."""
        state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'config': self.config,
            'fitted': self.fitted
        }
        joblib.dump(state, path)
        print(f"Preprocessor salvat în: {path}")
    
    def load_preprocessor(self, path: str):
        """Încarcă starea preprocessor-ului."""
        state = joblib.load(path)
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.config = state['config']
        self.fitted = state['fitted']
        print(f"Preprocessor încărcat din: {path}")


def save_splits_to_files(
    splits: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    base_path: str
) -> None:
    """
    Salvează seturile de date în fișiere.
    
    Args:
        splits: Dicționar cu seturile de date
        base_path: Calea de bază pentru salvare
    """
    for split_name, (X, y) in splits.items():
        split_path = os.path.join(base_path, split_name)
        os.makedirs(split_path, exist_ok=True)
        
        # Combinare și salvare
        data = pd.concat([X, y], axis=1)
        data.to_csv(os.path.join(split_path, f'{split_name}_data.csv'), index=False)
        
        print(f"Salvat: {split_path}/{split_name}_data.csv")


if __name__ == "__main__":
    # Exemplu de utilizare
    from data_acquisition.data_loader import generate_synthetic_data
    
    # Generare date test
    X, y = generate_synthetic_data(n_samples=1000, n_features=10)
    data = pd.concat([X, y], axis=1)
    
    # Preprocesare
    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess_pipeline(data, target_column='target')
    
    # Salvare
    save_splits_to_files(result['splits'], "../../data")
