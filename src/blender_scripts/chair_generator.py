"""Deterministic Blender script generator for a physically coherent parametric chair."""

from __future__ import annotations

import textwrap


def generate_chair_script(
    seat_height: float,
    seat_width: float,
    seat_depth: float,
    leg_count: int,
    leg_thickness: float,
    has_backrest: int,
    backrest_height: float,
    style_variant: int,
) -> str:
    """
    Generate a Blender (bpy) script that builds a parametric chair
    with correct physical contact between components.
    """

    # -------------------------
    # SAFETY / CLAMPS
    # -------------------------
    leg_count = max(3, min(4, int(leg_count)))
    has_backrest = 1 if int(has_backrest) == 1 else 0
    backrest_height = max(0.0, float(backrest_height))

    # Style variant (does NOT affect physical contact)
    style_offset = 0.0
    if style_variant == 1:
        style_offset = 0.02
    elif style_variant == 2:
        style_offset = -0.02

    script = f"""
import bpy

# -------------------------
# PARAMETERS (from UI / RN)
# -------------------------
seat_height = {seat_height}
seat_width = {seat_width}
seat_depth = {seat_depth}
leg_count = {leg_count}
leg_thickness = {leg_thickness}
has_backrest = {has_backrest}
backrest_height = {backrest_height}
style_offset = {style_offset}

# -------------------------
# CLEAN SCENE
# -------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# -------------------------
# DERIVED DIMENSIONS
# -------------------------
seat_thickness = max(0.03, seat_height * 0.2)
leg_height = max(0.01, seat_height - seat_thickness)
leg_radius = leg_thickness / 2.0

# -------------------------
# SEAT (rests ON legs)
# -------------------------
bpy.ops.mesh.primitive_cube_add(size=1)
seat = bpy.context.active_object
seat.name = "Seat"
seat.scale = (
    seat_width / 2.0,
    seat_depth / 2.0,
    seat_thickness / 2.0
)
seat.location = (
    0.0,
    0.0,
    leg_height + seat_thickness / 2.0
)

# -------------------------
# LEGS (touch seat exactly)
# Condition:
# |x| + leg_radius = seat_width / 2
# |y| + leg_radius = seat_depth / 2
# -------------------------
x_offset = (seat_width / 2.0) - leg_radius
y_offset = (seat_depth / 2.0) - leg_radius

leg_positions = [
    ( x_offset,  y_offset),
    (-x_offset,  y_offset),
    ( x_offset, -y_offset),
    (-x_offset, -y_offset),
][:leg_count]

legs = []

for idx, (x, y) in enumerate(leg_positions, start=1):
    bpy.ops.mesh.primitive_cylinder_add(radius=leg_radius, depth=leg_height)
    leg = bpy.context.active_object
    leg.name = f"Leg_{{idx}}"
    leg.location = (
        x,
        y,
        leg_height / 2.0
    )
    legs.append(leg)

# -------------------------
# BACKREST (touches seat)
# -------------------------
if has_backrest == 1 and backrest_height > 0:
    backrest_thickness = leg_thickness

    bpy.ops.mesh.primitive_cube_add(size=1)
    backrest = bpy.context.active_object
    backrest.name = "Backrest"
    backrest.scale = (
        seat_width / 2.0,
        backrest_thickness / 2.0,
        backrest_height / 2.0
    )
    backrest.location = (
        0.0,
        -(seat_depth / 2.0 + backrest_thickness / 2.0),
        leg_height + seat_thickness + backrest_height / 2.0
    )

# -------------------------
# OPTIONAL: JOIN ALL PARTS
# -------------------------
join_objects = True
if join_objects:
    bpy.ops.object.select_all(action='DESELECT')
    seat.select_set(True)
    for leg in legs:
        leg.select_set(True)
    if has_backrest == 1 and backrest_height > 0:
        backrest.select_set(True)

    bpy.context.view_layer.objects.active = seat
    bpy.ops.object.join()
"""

    return textwrap.dedent(script).strip() + "\\n"
