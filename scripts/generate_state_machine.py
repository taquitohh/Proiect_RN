"""
Script pentru generarea diagramei State Machine a sistemului Text-to-Blender.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_state_machine_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'State Machine - Text to Blender AI', 
            fontsize=16, fontweight='bold', ha='center', va='center')
    
    # Define states with positions
    states = {
        'IDLE': (2, 7, '#90EE90'),  # Light green
        'RECEIVE_INPUT': (5, 7, '#87CEEB'),  # Sky blue
        'VALIDATE_INPUT': (8, 7, '#87CEEB'),
        'CLASSIFY_INTENT': (11, 7, '#FFD700'),  # Gold - Neural Network
        'EXTRACT_PARAMS': (11, 5, '#FFD700'),
        'GENERATE_CODE': (8, 5, '#87CEEB'),
        'DISPLAY_OUTPUT': (5, 5, '#90EE90'),
        'ERROR_HANDLER': (5, 3, '#FF6B6B'),  # Red
    }
    
    # Draw states
    for state, (x, y, color) in states.items():
        # Draw box
        box = FancyBboxPatch((x-1.3, y-0.5), 2.6, 1, 
                              boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # State name
        display_name = state.replace('_', '\n')
        ax.text(x, y, display_name, fontsize=8, fontweight='bold', 
                ha='center', va='center')
    
    # Draw arrows (transitions)
    arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    kw = dict(arrowstyle=arrow_style, color="black", lw=1.5)
    
    # IDLE -> RECEIVE_INPUT
    ax.annotate("", xy=(3.7, 7), xytext=(2.7, 7),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(3.2, 7.3, 'user\ninput', fontsize=7, ha='center', va='bottom')
    
    # RECEIVE_INPUT -> VALIDATE_INPUT
    ax.annotate("", xy=(6.7, 7), xytext=(6.3, 7),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    
    # VALIDATE_INPUT -> CLASSIFY_INTENT
    ax.annotate("", xy=(9.7, 7), xytext=(9.3, 7),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(9.5, 7.3, 'valid', fontsize=7, ha='center', va='bottom', color='green')
    
    # CLASSIFY_INTENT -> EXTRACT_PARAMS
    ax.annotate("", xy=(11, 5.5), xytext=(11, 6.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(11.3, 6, 'intent\nfound', fontsize=7, ha='left', va='center')
    
    # EXTRACT_PARAMS -> GENERATE_CODE
    ax.annotate("", xy=(9.3, 5), xytext=(9.7, 5),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(9.5, 5.3, 'params', fontsize=7, ha='center', va='bottom')
    
    # GENERATE_CODE -> DISPLAY_OUTPUT
    ax.annotate("", xy=(6.3, 5), xytext=(6.7, 5),
                arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax.text(6.5, 5.3, 'code', fontsize=7, ha='center', va='bottom')
    
    # DISPLAY_OUTPUT -> IDLE (loop back)
    ax.annotate("", xy=(2, 6.5), xytext=(3.7, 5),
                arrowprops=dict(arrowstyle="->", color="green", lw=2,
                               connectionstyle="arc3,rad=0.3"))
    ax.text(2.5, 5.8, 'done', fontsize=7, ha='center', va='center', color='green')
    
    # VALIDATE_INPUT -> ERROR (invalid)
    ax.annotate("", xy=(6.3, 3), xytext=(8, 6.5),
                arrowprops=dict(arrowstyle="->", color="red", lw=2,
                               connectionstyle="arc3,rad=-0.2"))
    ax.text(7.5, 4.5, 'invalid', fontsize=7, ha='center', va='center', color='red')
    
    # GENERATE_CODE -> ERROR
    ax.annotate("", xy=(6.3, 3.3), xytext=(7.5, 4.5),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(7.2, 3.8, 'error', fontsize=7, ha='center', va='center', color='red')
    
    # ERROR -> IDLE (retry)
    ax.annotate("", xy=(2, 6.5), xytext=(3.7, 3),
                arrowprops=dict(arrowstyle="->", color="orange", lw=2,
                               connectionstyle="arc3,rad=0.4"))
    ax.text(2.2, 4.5, 'retry', fontsize=7, ha='center', va='center', color='orange')
    
    # Legend
    legend_y = 1.5
    legend_items = [
        ('#90EE90', 'Start/End States'),
        ('#87CEEB', 'Processing States'),
        ('#FFD700', 'AI/Neural Network'),
        ('#FF6B6B', 'Error Handler'),
    ]
    
    for i, (color, label) in enumerate(legend_items):
        rect = plt.Rectangle((1 + i*3.2, legend_y - 0.2), 0.4, 0.4, 
                             facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(1.5 + i*3.2, legend_y, label, fontsize=8, va='center')
    
    # Save
    plt.tight_layout()
    plt.savefig('e:/github/Proiect_RN/docs/state_machine.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print("Diagrama salvată în docs/state_machine.png")

if __name__ == "__main__":
    create_state_machine_diagram()
