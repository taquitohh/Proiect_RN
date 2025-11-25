"""
Modul Data Acquisition
======================

Funcții pentru achiziția și generarea datelor.
"""

from .data_loader import (
    load_csv_data,
    load_json_data,
    generate_synthetic_data,
    save_data,
    get_data_info,
    load_config
)

__all__ = [
    'load_csv_data',
    'load_json_data', 
    'generate_synthetic_data',
    'save_data',
    'get_data_info',
    'load_config'
]
