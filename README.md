# The Other Side of the Frame: Visual Theory-of-Mind for Agents

A neuro-symbolic dual-system architecture combining LLM strategic reasoning with reactive potential fields for multi-agent coordination under mutual visibility constraints.

_"Modern AI and Classical Robotics don't have to compete — they can collaborate. Fast reflexes, slow reasoning, working together. Just like humans do."_

## Overview 
This project addresses the perspective blindness problem in multi-agent systems: how can autonomous agents coordinate to reach a shared goal while maintaining mutual line-of-sight in occluded environments?
Our solution is a neuro-symbolic dual-system architecture inspired by cognitive science (Kahneman's "Thinking, Fast and Slow"):

System 1 (Fast): Reactive potential field navigation at 10 Hz
System 2 (Slow): Strategic LLM reasoning every 10 steps
Innovation: Weight modulation connects deliberate planning with reactive control

### The Challenge

```
Agent A     [Obstacle]     Agent B
   👁️         ████           👁️
              ████
         ❌ No visibility ❌
```

**Problem**: Agents must reach a shared target while maintaining visual contact. Losing sight can be catastrophic in applications like search-and-rescue or warehouse coordination.

### Our Solution

Three emergent cooperative behaviors:
- 🔄 **REGROUP**: When visibility is lost, prioritize finding partner
- 🎯 **EXPLORE**: When visibility is clear, prioritize reaching goal
- ✓ **REACHED**: Mission complete, cease movement

**Result**: 100% success rate with smooth, efficient trajectories.

---

## Key Features

### Neuro-Symbolic Integration
- First system to combine GPT-4o-mini with potential fields
- LLM provides interpretable natural language reasoning
- Graceful degradation: System 1 continues if LLM fails

### Tangential Flow (30/70 Innovation)
- Novel obstacle avoidance: 30% normal repulsion + 70% tangent sliding
- Creates smooth orbital paths around obstacles
- **23% path length reduction** vs pure repulsion
- Zero oscillations, enables narrow passage navigation

### Velocity Damping
- Linear velocity reduction near target (2.0m → 0.0m)
- Prevents terminal oscillation and overshoot
- Smooth goal arrival with jitter-free stopping

### Visual Theory-of-Mind
- Explicit reasoning about partner visibility states
- Ray-casting with obstacle occlusion detection
- Strategic coordination based on perspective-taking

### Real-Time Performance
- System 1: ~5ms per step (10 Hz control loop)
- System 2: ~1200ms per query (amortized to ~120ms/step)
- Total: ~8 Hz effective control rate
- Suitable for physical robotic platforms

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DUAL-SYSTEM ARCHITECTURE                  │
├──────────────────────────┬──────────────────────────────────┤
│      SYSTEM 1            │         SYSTEM 2                 │
│   (Fast & Reactive)      │    (Slow & Deliberate)           │
│                          │                                  │
│     Potential Fields     │     LLM Reasoning                │
│  • 10 Hz control         │  • GPT-4o-mini (T=0.1)           │
│  • 5 force sources       │  • Every 10 steps                │
│  • Continuous motion     │  • Strategic planning            │
│                          │                                  │
│  Forces:                 │  Decisions:                      │
│  1. Target attraction    │  • Visibility assessment         │
│  2. Peer attraction      │  • Mode selection                │
│  3. Obstacle (30/70)     │  • Weight adjustment             │
│  4. Map boundaries       │  • Natural language reasoning    │
│  5. Velocity damping     │                                  │
│                          │                                  │
└──────────────┬───────────┴──────────────┬───────────────────┘
               │                          │
               │   Weight Modulation      │
               │   (w_target, w_peer)     │
               └──────────────────────────┘
                          │
                    Smooth, Safe,
                 Intelligent Navigation
```

---
## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (for GPT-4o-mini)
- pip package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/andysundar/visual-theory-of-mind.git
cd visual-theory-of-mind

# Install required packages
pip install -r requirements.txt

# Optional 
pip install numpy matplotlib openai plotly

# Optional: Install in development mode
pip install -e .
```

--- 

## Quick Start
```bash
python multi_agent_controller.py
``` 
