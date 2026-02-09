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

## Fridge dataset (fridges_dataset.csv)

- Location: data/generated/fridges_dataset.csv
- Samples: 12000
- Features (8):
  - fridge_height, fridge_width, fridge_depth
  - door_thickness, handle_length
  - freezer_ratio, freezer_position
  - style_variant
- Labels (2):
  - 0 = top_freezer
  - 1 = bottom_freezer

## Stove dataset (stoves_dataset.csv)

- Location: data/generated/stoves_dataset.csv
- Samples: 12000
- Features (9):
  - stove_height, stove_width, stove_depth
  - oven_height_ratio
  - handle_length, glass_thickness
  - style_variant
  - burner_count, knob_count
- Labels (3):
  - 0 = gas
  - 1 = electric
  - 2 = induction

## Processed outputs

- Cleaned data: data/processed/*_clean.csv
- Scaled data: data/processed/*_scaled.csv
- Split sets:
  - Chairs: data/train, data/validation, data/test
  - Tables: data/tables/train, data/tables/validation, data/tables/test
  - Cabinets: data/cabinets/train, data/cabinets/validation, data/cabinets/test
  - Fridges: data/fridges/train, data/fridges/validation, data/fridges/test
  - Stoves: data/stoves/train, data/stoves/validation, data/stoves/test
