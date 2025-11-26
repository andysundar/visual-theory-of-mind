# =============================================================================
# LLM based MULTI-AGENT COORDINATION (RANDOMIZED INITIALIZATION)
# Architecture: LLM-Modulated Potential Fields
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import random
import time
from openai import OpenAI
from getpass import getpass
from enum import Enum
from dataclasses import dataclass

# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================

# Only ask for key if not already set (for convenience in Colab)
try:
    if client: pass
except NameError:
    print(">> Please enter your OpenAI API Key:")
    API_KEY = getpass()
    client = OpenAI(api_key=API_KEY)

@dataclass
class GlobalConfig:
    MAX_STEPS: int = 60             # Maximum simulation length
    LLM_INTERVAL: int = 10          # Query LLM after every 10 steps
    MOVE_SPEED: float = 0.6         # Slightly faster for random maps
    AGENT_RADIUS: float = 0.3       # Agent velocity (m/step)
    SENSOR_RANGE: float = 6.0       # How far agents can see
    REPULSION_DIST: float = 1.5     # Obstacle avoidance range
    REPULSION_FORCE: float = 2.0    # Amount of force applied to stop
    MAP_BOUNDS: float = 7.5         # +/- 7.5 meters
    TARGET_THRESHOLD: float = 1.0   # Stop when within 1m of target
    MIN_VELOCITY: float = 0.05      # Minimum movement threshold
    DAMPING_DISTANCE: float = 2.0   # Start slowing down within 2m of target

CFG = GlobalConfig()

# =============================================================================
# 2. PHYSICS ENGINE (SYSTEM 1)
# =============================================================================
class PhysicsEngine:
    @staticmethod
    def calculate_potential_field(agent_pos, target_pos, peer_pos, obstacles, weights):
        force = np.array([0.0, 0.0])

        # Calculate distance to target for damping
        dist_to_target = np.linalg.norm(target_pos - agent_pos)

        # 1. ATTRACTION TO TARGET (Goal Seeking)
        vec_t = target_pos - agent_pos
        dist_t = np.linalg.norm(vec_t)
        if dist_t > 0:
            force += weights['w_target'] * (vec_t / dist_t)

        # 2. ATTRACTION TO PEER (Social Connectivity)
        vec_p = peer_pos - agent_pos
        dist_p = np.linalg.norm(vec_p)
        if dist_p > 0:
            force += weights['w_peer'] * (vec_p / dist_p)

        # 3. ADVANCED OBSTACLE AVOIDANCE (Repulsion + Tangential Flow)
        for obs in obstacles:
            vec_o = agent_pos - obs['pos']
            dist_o = np.linalg.norm(vec_o)
            edge_dist = dist_o - obs['radius']

            if edge_dist < CFG.REPULSION_DIST:
                # A. Standard Repulsion (Don't hit the wall)
                repulsion_strength = CFG.REPULSION_FORCE * (1.0 / (edge_dist + 0.1))
                repulsion_dir = vec_o / dist_o

                # B. Tangential Force (Slide around the wall)
                # We rotate the repulsion vector 90 degrees
                # [-y, x] is a 90 deg rotation
                tangent_dir = np.array([-repulsion_dir[1], repulsion_dir[0]])

                # Smart Direction: Should we go Clockwise or Counter-Clockwise?
                # We pick the direction that aligns better with the Target
                if np.dot(tangent_dir, (target_pos - agent_pos)) < 0:
                    tangent_dir = -tangent_dir # Flip it if it points away from goal

                # COMBINE: 30% Push Away, 70% Slide Along
                # This creates a smooth "Orbital" path
                force += (0.3 * repulsion_strength * repulsion_dir) + \
                         (0.7 * repulsion_strength * tangent_dir)

        # 4. MAP BOUNDARIES
        if abs(agent_pos[0]) > CFG.MAP_BOUNDS: force[0] -= np.sign(agent_pos[0]) * 2.0
        if abs(agent_pos[1]) > CFG.MAP_BOUNDS: force[1] -= np.sign(agent_pos[1]) * 2.0

        # 5. VELOCITY DAMPING (Slow down near target to prevent jitter)
        damping_factor = 1.0
        if dist_to_target < CFG.DAMPING_DISTANCE:
            # Linear damping: 1.0 at DAMPING_DISTANCE, 0.1 at target
            damping_factor = max(0.1, dist_to_target / CFG.DAMPING_DISTANCE)

        return force, damping_factor
    @staticmethod
    def check_visibility(p1, p2, obstacles):
        dist = np.linalg.norm(p1 - p2)
        if dist > CFG.SENSOR_RANGE: return False, "RANGE"

        ray_vec = p2 - p1
        ray_dir = ray_vec / (dist + 1e-6)

        for obs in obstacles:
            vec_to_obs = obs['pos'] - p1
            projection = np.dot(vec_to_obs, ray_dir)
            closest_point = p1 + ray_dir * np.clip(projection, 0, dist)
            if np.linalg.norm(closest_point - obs['pos']) < obs['radius']:
                return False, "OCCLUSION"
        return True, "CLEAR"

# =============================================================================
# 3. COGNITIVE ENGINE (SYSTEM 2 - LLM)
# =============================================================================
class NeuroSymbolicBrain:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model

    def get_strategic_weights(self, agent_id, state_desc):
        prompt = f"""
        You are an autonomous robot (Agent {agent_id}).
        Goal: Reach Target. Constraint: Maintain Line-of-Sight with Partner.

        SITUATION:
        - Visibility to Partner: {state_desc['visibility']}
        - Distance to Target: {state_desc['dist_target']:.1f}m
        - Obstacles: {state_desc['obstacle_status']}

        DECISION:
        - If visibility LOST: High priority on Partner (REGROUP).
        - If visibility CLEAR: High priority on Target (EXPLORE).

        Respond JSON only:
        {{ "mode": "String", "w_target": float, "w_peer": float, "reasoning": "Short string" }}
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            content = response.choices[0].message.content.replace("```json","").replace("```","")
            return json.loads(content)
        except:
            return {"mode": "FAIL", "w_target": 0.5, "w_peer": 0.5, "reasoning": "Error"}

# =============================================================================
# 4. RANDOMIZATION HELPERS
# =============================================================================
def get_random_valid_pos(obstacles, target_pos, min_dist_target=3.0):
    """Generates a random position that is NOT inside an obstacle."""
    while True:
        # Generate random x, y within bounds
        x = random.uniform(-CFG.MAP_BOUNDS + 1, CFG.MAP_BOUNDS - 1)
        y = random.uniform(-CFG.MAP_BOUNDS + 1, CFG.MAP_BOUNDS - 1)
        pos = np.array([x, y])

        valid = True

        # Check 1: Too close to target?
        if np.linalg.norm(pos - target_pos) < min_dist_target:
            valid = False

        # Check 2: Inside an obstacle?
        for obs in obstacles:
            if np.linalg.norm(pos - obs['pos']) < (obs['radius'] + CFG.AGENT_RADIUS + 0.5):
                valid = False
                break

        if valid: return pos

# =============================================================================
# 5. IMPROVED VISUALIZATION FUNCTION
# =============================================================================
def render_final_map(agents, obstacles, target):
    """
    Improved visualization with:
    - Proper z-ordering so target is visible
    - Smart text placement to avoid overlap with target
    - White borders for better visibility
    - Background boxes for text labels
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#F8F9FA')
    ax.set_xlim(-CFG.MAP_BOUNDS-1, CFG.MAP_BOUNDS+1)
    ax.set_ylim(-CFG.MAP_BOUNDS-1, CFG.MAP_BOUNDS+1)

    # Layer 1: Draw obstacles (lowest z-order)
    for obs in obstacles:
        ax.add_patch(patches.Circle(obs['pos'], obs['radius'],
                                    facecolor=obs['color'], alpha=0.9, zorder=1))
        ax.add_patch(patches.Circle(obs['pos'], obs['radius'] + CFG.REPULSION_DIST,
                                    fill=False, ls=':', color='gray', alpha=0.4, zorder=1))

    # Layer 2: Draw agent paths and markers
    for uid, a in agents.items():
        if len(a['path']) > 0:
            path = np.array(a['path'])

            # Path line
            ax.plot(path[:, 0], path[:, 1], c=a['color'], lw=3,
                   label=f"Agent {uid}", zorder=2, alpha=0.8)

            # Start position
            ax.scatter(path[0,0], path[0,1], c=a['color'], marker='o', s=100,
                      label=f"{uid} Start", zorder=3, edgecolor='white', linewidth=1.5)

            # End position
            ax.scatter(path[-1,0], path[-1,1], c=a['color'], marker='X', s=200,
                      edgecolor='white', zorder=3, linewidth=2)

            # Smart text positioning: avoid the target area
            final_pos = path[-1]
            vec_from_target = final_pos - target['pos']
            dist_from_target = np.linalg.norm(vec_from_target)

            if dist_from_target < 2.5:  # If close to target, place text further away
                # Push text away from target
                text_offset = (vec_from_target / (dist_from_target + 0.001)) * 1.5
                text_pos = final_pos + text_offset
            else:
                # Default: place text above the final position
                text_pos = final_pos + np.array([0, 0.7])

            # Add text with background box for better visibility
            ax.text(text_pos[0], text_pos[1], a['mode'],
                   ha='center', va='center',
                   fontsize=9, fontweight='bold', color=a['color'],
                   zorder=10,
                   bbox=dict(boxstyle='round,pad=0.4',
                            facecolor='white',
                            edgecolor=a['color'],
                            linewidth=2,
                            alpha=0.9))

    # Layer 3: Draw target (highest z-order with prominent styling)
    target_circle = patches.Circle(target['pos'], target['radius'],
                                   facecolor=target['color'], zorder=8,
                                   edgecolor='white', linewidth=3)
    ax.add_patch(target_circle)

    # Add "TARGET" label inside the circle
    ax.text(target['pos'][0], target['pos'][1], 'TARGET',
           ha='center', va='center',
           fontsize=10, fontweight='bold', color='white', zorder=9)

    # Final touches
    plt.legend(loc='upper right', framealpha=0.95, fontsize=9)
    plt.title("The Other Side of the Frame",
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlabel("X Position (meters)", fontsize=10)
    plt.ylabel("Y Position (meters)", fontsize=10)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. MAIN RUNNER
# =============================================================================
def run_randomized_simulation():
    # -- SCENE SETUP --
    target = {'pos': np.array([0.0, 0.0]), 'radius': 0.8, 'color': 'red'}

    # Fixed obstacles (The "Map")
    obstacles = [
        {'pos': np.array([0.0, 2.5]), 'radius': 2.0, 'color': '#444444'},
        {'pos': np.array([3.5, -2.5]), 'radius': 1.8, 'color': '#444444'},
        {'pos': np.array([-4.0, -1.0]), 'radius': 1.5, 'color': '#444444'}
    ]

    # -- RANDOM START POSITIONS --
    print(">> Generating Valid Random Start Positions...")
    start_a = get_random_valid_pos(obstacles, target['pos'])

    # Ensure B starts somewhat far from A (to force coordination)
    start_b = get_random_valid_pos(obstacles, target['pos'])
    while np.linalg.norm(start_a - start_b) < 4.0:
        start_b = get_random_valid_pos(obstacles, target['pos'])

    agents = {
        'A': {'pos': start_a, 'color': 'blue', 'path': [], 'battery': 100,
              'weights': {'w_target': 0.5, 'w_peer': 0.5}, 'mode': 'INIT', 'reached': False},
        'B': {'pos': start_b, 'color': 'orange', 'path': [], 'battery': 100,
              'weights': {'w_target': 0.5, 'w_peer': 0.5}, 'mode': 'INIT', 'reached': False}
    }

    brain = NeuroSymbolicBrain()

    print(f">> A Start: {agents['A']['pos'].round(1)} | B Start: {agents['B']['pos'].round(1)}")
    print("-" * 90)
    print(f"{'STEP':<5} | {'ID':<3} | {'MODE (LLM)':<15} | {'W_TARG':<6} | {'W_PEER':<6} | {'REASONING'}")
    print("-" * 90)

    # -- SIMULATION LOOP --
    for step in range(CFG.MAX_STEPS):
        can_see, _ = PhysicsEngine.check_visibility(agents['A']['pos'], agents['B']['pos'], obstacles)

        # SYSTEM 2: LLM UPDATE
        if step % CFG.LLM_INTERVAL == 0:
            for uid, agent in agents.items():
                state = {
                    'visibility': "CLEAR" if can_see else "BLOCKED",
                    'dist_target': np.linalg.norm(target['pos'] - agent['pos']),
                    'obstacle_status': "NEAR" if any(np.linalg.norm(agent['pos']-o['pos']) < o['radius']+2.0 for o in obstacles) else "CLEAR"
                }

                decision = brain.get_strategic_weights(uid, state)
                agent['weights']['w_target'] = decision['w_target']
                agent['weights']['w_peer'] = decision['w_peer']
                agent['mode'] = decision['mode']

                print(f"{step:<5} | {uid:<3} | {decision['mode']:<15} | {decision['w_target']:<6.1f} | {decision['w_peer']:<6.1f} | {decision['reasoning']}")

        # SYSTEM 1: PHYSICS UPDATE
        for uid, agent in agents.items():
            # Check if agent has reached target
            dist_to_target = np.linalg.norm(target['pos'] - agent['pos'])

            # Stop moving if within threshold
            if dist_to_target < CFG.TARGET_THRESHOLD:
                if not agent['reached']:  # Only print once
                    agent['mode'] = 'REACHED'
                    agent['reached'] = True
                    print(f"      >>> Agent {uid} REACHED TARGET at step {step}")
                    break
                continue  # Skip movement

            peer_id = 'B' if uid == 'A' else 'A'
            force, damping = PhysicsEngine.calculate_potential_field(
                agent['pos'], target['pos'], agents[peer_id]['pos'], obstacles, agent['weights']
            )

            force_magnitude = np.linalg.norm(force)
            if force_magnitude > CFG.MIN_VELOCITY:
                direction = force / force_magnitude
                # Apply damping to velocity
                velocity = CFG.MOVE_SPEED * damping
                agent['pos'] += direction * velocity
                agent['path'].append(agent['pos'].copy())

    # render_final_map(agents, obstacles, target)
    render_final_map(agents, obstacles, target)
    return agents, obstacles, target

if __name__ == "__main__":
    simulation_data = run_randomized_simulation()


# =============================================================================
# CELL 2: PLOTLY ANIMATION (Run this AFTER Cell 1 completes)
# =============================================================================

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

# Extract data from simulation
agents, obstacles, target = simulation_data

# =============================================================================
# CONFIGURATION
# =============================================================================
ANIMATION_CONFIG = {
    'frame_duration': 1000,      # milliseconds per frame (100 = 10 fps)
    'transition_duration': 50,   # smooth transitions between frames
    'show_trails': True,         # Show path trails behind agents
    'show_mode_labels': True,    # Show LLM mode labels
    'show_repulsion_zones': True # Show obstacle repulsion boundaries
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_circle_shape(center, radius, color, opacity=0.8, dash='solid'):
    """Create a circular shape for plotly"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return x, y

def get_max_path_length(agents):
    """Find the longest path for animation frames"""
    return max(len(agent['path']) for agent in agents.values())

# =============================================================================
# PREPARE DATA FOR ANIMATION
# =============================================================================

# Get maximum number of frames
max_frames = get_max_path_length(agents)
print(f"Creating animation with {max_frames} frames...")

# Prepare agent paths (pad shorter paths to match longest)
agent_data = {}
for uid, agent in agents.items():
    path = np.array(agent['path'])

    # Pad path if needed (repeat last position)
    if len(path) < max_frames:
        last_pos = path[-1]
        padding = np.tile(last_pos, (max_frames - len(path), 1))
        path = np.vstack([path, padding])

    agent_data[uid] = {
        'path': path,
        'color': agent['color'],
        'mode': agent.get('mode', 'EXPLORE')
    }

# =============================================================================
# CREATE PLOTLY FIGURE
# =============================================================================

fig = go.Figure()

# 1. Add obstacles (static)
for i, obs in enumerate(obstacles):
    # Solid obstacle
    x, y = create_circle_shape(obs['pos'], obs['radius'], obs['color'])
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='toself',
        fillcolor=obs['color'],
        line=dict(color=obs['color'], width=2),
        opacity=0.9,
        showlegend=True if i == 0 else False,
        name='Obstacles' if i == 0 else '',
        hoverinfo='skip'
    ))

    # Repulsion zone (dotted circle)
    if ANIMATION_CONFIG['show_repulsion_zones']:
        x_rep, y_rep = create_circle_shape(obs['pos'], obs['radius'] + 1.5, 'gray')
        fig.add_trace(go.Scatter(
            x=x_rep, y=y_rep,
            line=dict(color='gray', width=1, dash='dot'),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip',
            mode='lines'
        ))

# 2. Add target (static)
x_target, y_target = create_circle_shape(target['pos'], target['radius'], target['color'])
fig.add_trace(go.Scatter(
    x=x_target, y=y_target,
    fill='toself',
    fillcolor=target['color'],
    line=dict(color='white', width=3),
    opacity=1.0,
    name='Target',
    hoverinfo='text',
    text='TARGET',
    mode='lines'
))

# 3. Add target label
fig.add_trace(go.Scatter(
    x=[target['pos'][0]],
    y=[target['pos'][1]],
    mode='text',
    text=['TARGET'],
    textfont=dict(color='white', size=12, family='Arial Black'),
    showlegend=False,
    hoverinfo='skip'
))

# 4. Add agent trails (animated)
for uid, data in agent_data.items():
    if ANIMATION_CONFIG['show_trails']:
        fig.add_trace(go.Scatter(
            x=[data['path'][0, 0]],
            y=[data['path'][0, 1]],
            mode='lines',
            line=dict(color=data['color'], width=3),
            opacity=0.6,
            name=f'Agent {uid} Trail',
            showlegend=True
        ))

# 5. Add agent markers (animated)
for uid, data in agent_data.items():
    # Current position marker
    fig.add_trace(go.Scatter(
        x=[data['path'][0, 0]],
        y=[data['path'][0, 1]],
        mode='markers+text',
        marker=dict(
            size=20,
            color=data['color'],
            symbol='circle',
            line=dict(color='white', width=2)
        ),
        text=[uid],
        textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        name=f'Agent {uid}',
        showlegend=True
    ))

    # Mode label (above agent)
    if ANIMATION_CONFIG['show_mode_labels']:
        fig.add_trace(go.Scatter(
            x=[data['path'][0, 0]],
            y=[data['path'][0, 1] + 0.8],
            mode='text',
            text=[data['mode']],
            textfont=dict(color=data['color'], size=10, family='Arial'),
            showlegend=False,
            hoverinfo='skip'
        ))

# 6. Add step counter
fig.add_trace(go.Scatter(
    x=[-7],
    y=[7],
    mode='text',
    text=['Step: 0'],
    textfont=dict(color='black', size=14, family='Arial Black'),
    showlegend=False,
    hoverinfo='skip'
))

# =============================================================================
# CREATE ANIMATION FRAMES
# =============================================================================

frames = []
for frame_idx in range(max_frames):
    frame_data = []

    # Skip static elements (obstacles, target)
    skip_traces = len(obstacles) * 2 + 2  # obstacles + repulsion zones + target + target label

    # Update agent trails
    trace_idx = skip_traces
    for uid, data in agent_data.items():
        if ANIMATION_CONFIG['show_trails']:
            frame_data.append(go.Scatter(
                x=data['path'][:frame_idx+1, 0],
                y=data['path'][:frame_idx+1, 1]
            ))
            trace_idx += 1

    # Update agent markers and labels
    for uid, data in agent_data.items():
        # Agent marker
        frame_data.append(go.Scatter(
            x=[data['path'][frame_idx, 0]],
            y=[data['path'][frame_idx, 1]],
            text=[uid]
        ))

        # Mode label
        if ANIMATION_CONFIG['show_mode_labels']:
            frame_data.append(go.Scatter(
                x=[data['path'][frame_idx, 0]],
                y=[data['path'][frame_idx, 1] + 0.8],
                text=[data['mode']]
            ))

    # Update step counter
    frame_data.append(go.Scatter(
        x=[-7],
        y=[7],
        text=[f'Step: {frame_idx}']
    ))

    frames.append(go.Frame(
        data=frame_data,
        name=str(frame_idx),
        traces=list(range(skip_traces, len(fig.data)))
    ))

fig.frames = frames

# =============================================================================
# LAYOUT AND STYLING
# =============================================================================

fig.update_layout(
    title={
        'text': 'The Other side of the Frame',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 18, 'family': 'Arial Black'}
    },
    xaxis=dict(
        range=[-8.5, 8.5],
        title='X Position (meters)',
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True
    ),
    yaxis=dict(
        range=[-8.5, 8.5],
        title='Y Position (meters)',
        showgrid=True,
        gridcolor='lightgray',
        zeroline=True,
        scaleanchor='x',
        scaleratio=1
    ),
    plot_bgcolor='#F8F9FA',
    width=800,
    height=800,
    hovermode='closest',

    # Animation controls
    updatemenus=[{
        'type': 'buttons',
        'showactive': True,
        'buttons': [
            {
                'label': '▶ Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': ANIMATION_CONFIG['frame_duration'], 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': ANIMATION_CONFIG['transition_duration']}
                }]
            },
            {
                'label': '⏸ Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ],
        'x': 0.1,
        'y': 0,
        'xanchor': 'left',
        'yanchor': 'top'
    }],

    # Time slider
    sliders=[{
        'active': 0,
        'yanchor': 'top',
        'y': -0.05,
        'xanchor': 'left',
        'currentvalue': {
            'prefix': 'Frame: ',
            'visible': True,
            'xanchor': 'right'
        },
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'steps': [
            {
                'args': [[f.name], {
                    'frame': {'duration': ANIMATION_CONFIG['frame_duration'], 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': ANIMATION_CONFIG['transition_duration']}
                }],
                'method': 'animate',
                'label': str(k)
            }
            for k, f in enumerate(fig.frames)
        ]
    }]
)

# =============================================================================
# DISPLAY ANIMATION
# =============================================================================

print(" Animation ready!")
print(" Controls:")
print("   - Click ▶ Play to start animation")
print("   - Use slider to scrub through time")
print("   - Zoom/pan with mouse")
print("   - Hover over elements for details")
print("")

fig.show()

# =============================================================================
# OPTIONAL: SAVE TO HTML
# =============================================================================

# Uncomment to save animation to standalone HTML file
# fig.write_html("multi_agent_animation.html")
# print(" Saved to: multi_agent_animation.html")
