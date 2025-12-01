# Project Armstrong: AI Agents for Lunar Landing By Deependra Shekhawat

![Project Armstrong](mission_moon.jpg)

## Overview
This project implements a multi-agent AI system to pilot a Lunar Lander safely to the surface. It uses Google's Gemini 2.5 Flash-Lite model via the **Agent Development Kit (ADK)** to power two distinct agents: a **Navigator** and a **Commander**.

This project is designed for the **Kaggle Agents Intensive Capstone**.

## Mission Replay

<video src="mission_replay.mp4" controls width="600"></video>

*Watch the AI agents navigate and land the lunar lander in real-time.*

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

### High-Level Design
The system uses a hierarchical multi-agent architecture with ADK components:

1.  **Navigator Agent** (LlmAgent):
    *   **Role**: Strategic Advisor.
    *   **Input**: Raw telemetry (Altitude, Velocity, Angle).
    *   **Memory**: Maintains session-based conversation history via `InMemorySessionService`.
    *   **Context Management**: Uses `EventsCompactionConfig` to prevent unbounded history growth.
    *   **Output**: High-level advice (e.g., "Stabilize angle before descending").

2.  **Commander Agent** (LlmAgent):
    *   **Role**: Pilot / Executor.
    *   **Input**: Telemetry + Navigator's Advice.
    *   **Mechanism**: Uses **Function Calling** (`execute_maneuver` tool) to interact with the environment.
    *   **Output**: Specific actions (Main Engine, Side Engines) and durations.

3.  **ADK Integration**:
    *   **Runner**: Orchestrates agent execution with session management
    *   **App**: Wraps Navigator with context compaction configuration
    *   **SessionService**: Manages conversation state across turns
    *   **Gemini Model**: Powered by gemini-2.5-flash-lite with retry logic

## Key Features
*   **Multi-Agent System**: Hierarchical coordination between Navigator (strategy) and Commander (tactics)
*   **ADK Framework**: Built using Google's Agent Development Kit following course best practices
*   **Session Management**: Navigator maintains conversation history with context compaction
*   **Function Calling**: Commander uses ADK function tools for environment interaction
*   **Observability**: Tracks token usage, latency, and errors for every LLM call
*   **Robustness**: Implements exponential backoff retry logic for API stability
*   **Physics-Aware Prompts**: Agents are primed with specific physics rules (e.g., "Stabilize First") to prevent common failure modes like the "Death Spiral"
*   **Evaluation**: Includes a script to run multiple episodes and calculate metrics

## Files
*   `agents.py`: Defines agent creation functions (`create_navigator_agent`, `create_commander_agent`) using ADK's LlmAgent
*   `main_mission.py`: Runs a single demo mission with video recording using ADK Runner
*   `evaluate_agent.py`: Runs a batch of episodes to calculate metrics with ADK patterns
*   `lunar_tools.py`: Interface for the Gymnasium LunarLander-v2 environment
*   `ARCHITECTURE.md`: Detailed architecture documentation with diagrams

## How to Run

### Prerequisites
- Python 3.10+ (recommended) or Python 3.9 (with warnings)
- Conda environment manager
- **Visual Studio C++ Build Tools** (required for Box2D compilation on Windows)
  - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Or install via: `winget install Microsoft.VisualStudio.2022.BuildTools`

### 1. Setup Environment with Conda
```bash
# Create conda environment with Python 3.10+
conda create -n armstrong_env python=3.10
conda activate armstrong_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Install Box2D (if needed)
```bash
# Try this first - requires Visual Studio Build Tools
pip install gymnasium[box2d]

# If compilation fails, install Build Tools then retry
```

### 3. Set API Key
Set your `GOOGLE_API_KEY` in a `.env` file or environment variable:
```bash
# In PowerShell
$env:GOOGLE_API_KEY='your_key_here'

# Or create .env file with:
GOOGLE_API_KEY=your_key_here
```

### 4. Run a Mission
```bash
# Make sure you're in the conda environment
python main_mission.py
```

### 5. Run Evaluation
```bash
python evaluate_agent.py
```

## Implementation Notes

### ADK Integration Status
✅ **Completed**:
- Agents use `LlmAgent` from `google.adk.agents`
- Session management via `InMemorySessionService`
- Context compaction with `EventsCompactionConfig` (10 turns, 2 overlap)
- Function tools following ADK best practices
- Runner orchestration for Navigator agent
- Async/await patterns for ADK compatibility

⚠️ **Known Issues**:
- Box2D requires Visual Studio C++ Build Tools on Windows
- Python 3.9 shows deprecation warnings (upgrade to 3.10+ recommended)

### Rate Limiting
The code includes 9-second delays between steps to respect Gemini API rate limits (15 RPM for free tier with 2 calls per step).

## Results
The agents demonstrate the ability to:
*   Recover from unstable initial conditions.
*   Coordinate strategy (Navigator) and tactics (Commander).
*   Land safely (Reward > 200) in successful episodes.
