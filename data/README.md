# Dataset overview

This project uses synthetic, deterministic datasets for multiple object types.
All datasets are generated programmatically with fixed seeds for reproducibility.

## Chair dataset (chairs_dataset.csv)

- Location: data/generated/chairs_dataset.csv
- Samples: 15000
- Features (8):
  - seat_height, seat_width, seat_depth
  - leg_count, leg_thickness
  - has_backrest, backrest_height
  - style_variant
- Labels (4):
  - 0 = simple chair
  - 1 = chair with backrest
  - 2 = bar chair
  - 3 = stool

## Table dataset (tables_dataset.csv)

- Location: data/generated/tables_dataset.csv
- Samples: 12000
- Features (7):
  - table_height, table_width, table_depth
  - leg_count, leg_thickness
  - has_apron
  - style_variant
- Labels (3):
  - 0 = low_table
  - 1 = dining_table
  - 2 = bar_table

## Cabinet dataset (cabinets_dataset.csv)

- Location: data/generated/cabinets_dataset.csv
- Samples: 12000
- Features (7):
  - cabinet_height, cabinet_width, cabinet_depth
  - wall_thickness
  - door_type, door_count
  - style_variant
- Labels (3):
  - 0 = single_door
  - 1 = double_door
  - 2 = tall_cabinet

## Processed outputs

- Cleaned data: data/processed/*_clean.csv
- Scaled data: data/processed/*_scaled.csv
- Split sets:
  - Chairs: data/train, data/validation, data/test
  - Tables: data/tables/train, data/tables/validation, data/tables/test
  - Cabinets: data/cabinets/train, data/cabinets/validation, data/cabinets/test
