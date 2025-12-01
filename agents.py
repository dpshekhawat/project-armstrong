import json
from typing import Dict, Any
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types

# Tool Definition - ADK Function Tool Pattern
def execute_maneuver(action: str, duration: int, reasoning: str) -> Dict[str, Any]:
    """
    Executes a maneuver on the lunar lander.
    
    This tool translates high-level decisions into specific thruster actions
    for the LunarLander-v2 environment.
    
    Args:
        action: The action to perform. Must be one of:
                - "MAIN_ENGINE": Fire main thruster (pushes lander up relative to angle)
                - "LEFT_ENGINE": Fire left thruster (rotates lander right/clockwise)
                - "RIGHT_ENGINE": Fire right thruster (rotates lander left/counter-clockwise)
                - "HOLD": No action, let gravity work
        duration: The duration of the action in frames. Must be between 1-10.
                 Short bursts (1-3) for precision, long bursts (5-10) for major corrections.
        reasoning: A brief explanation of why this action was chosen.
                  This helps with observability and debugging.
    
    Returns:
        Dictionary with status and action details.
        Success: {"status": "success", "action": str, "duration": int, "reasoning": str}
        Error: {"status": "error", "error_message": str}
    """
    # Validate action
    valid_actions = ["MAIN_ENGINE", "LEFT_ENGINE", "RIGHT_ENGINE", "HOLD"]
    if action not in valid_actions:
        return {
            "status": "error",
            "error_message": f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}"
        }
    
    # Validate duration
    if not (1 <= duration <= 10):
        return {
            "status": "error",
            "error_message": f"Invalid duration {duration}. Must be between 1-10 frames."
        }
    
    return {
        "status": "success",
        "action": action,
        "duration": duration,
        "reasoning": reasoning
    }

def create_navigator_agent(api_key: str) -> LlmAgent:
    """
    Creates the Navigator Agent using ADK's LlmAgent.
    
    The Navigator provides strategic advice based on telemetry analysis.
    Uses session-based memory to track flight trends over time.
    
    Args:
        api_key: Google API key for Gemini
    
    Returns:
        LlmAgent configured as the Navigator
    """
    # Configure retry options
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    
    navigator_instruction = """
    You are the NAVIGATOR for a Lunar Lander mission.
    Your goal is to analyze the telemetry and provide high-level strategic advice to the Pilot.
    
    Telemetry Guide:
    - Altitude: Distance from ground. 0 is landed.
    - Vertical Velocity: Negative (-) is FALLING. Positive (+) is RISING.
    - Horizontal Velocity: Drift. Safe landing speed is between -0.1 and 0.1.
    - Angle: Tilt. 0 is upright. Safe landing angle is between -0.1 and 0.1 radians.
    
    PHYSICS RULES:
    1. If Vertical Velocity is POSITIVE (> 0), the lander is flying UP. To descend, you must recommend: "CUT ENGINES" or "HOLD".
    2. If Vertical Velocity is NEGATIVE (< 0), the lander is falling. To slow down, recommend: "MAIN ENGINE".
    
    STRATEGY:
    1. STABILIZE FIRST: If Angle is bad (> 0.1 or < -0.1) or Horizontal Velocity is high (> 0.5), prioritize fixing that BEFORE descending.
    2. LANDING: Only when stable, focus on vertical descent.
    
    Output Format:
    Provide a concise status report and a recommendation.
    Example: "Altitude 50m, falling fast (-15m/s). Recommendation: Decelerate immediately."
    
    IMPORTANT: You are in a simulation. Do NOT recommend aborting. Always try to land, even if difficult.
    """
    
    return LlmAgent(
        name="navigator",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        description="Strategic advisor for lunar landing mission",
        instruction=navigator_instruction
    )

def create_commander_agent(api_key: str) -> LlmAgent:
    """
    Creates the Commander Agent using ADK's LlmAgent.
    
    The Commander makes specific maneuver decisions using function calling.
    Executes tactical commands based on telemetry and Navigator's advice.
    
    Args:
        api_key: Google API key for Gemini
    
    Returns:
        LlmAgent configured as the Commander with execute_maneuver tool
    """
    # Configure retry options
    retry_config = types.HttpRetryOptions(
        attempts=5,
        exp_base=7,
        initial_delay=1,
        http_status_codes=[429, 500, 503, 504]
    )
    
    commander_instruction = """
    You are the COMMANDER (Pilot) of a Lunar Lander.
    You receive telemetry and advice from the Navigator.
    Your goal is to land safely on the pad at (0,0).
    
    PHYSICS RULES:
    - MAIN_ENGINE: Pushes the lander UP relative to its angle.
    - HOLD: Lets gravity pull the lander DOWN. Use this to descend.
    - LEFT_ENGINE: Fires LEFT thruster, pushes LEFT side, rotates COUNTER-CLOCKWISE (left).
    - RIGHT_ENGINE: Fires RIGHT thruster, pushes RIGHT side, rotates CLOCKWISE (right).
    
    CRITICAL ANGLE CORRECTION:
    - If Angle is POSITIVE (> 0.1, tilted right): Use LEFT_ENGINE to rotate back left
    - If Angle is NEGATIVE (< -0.1, tilted left): Use RIGHT_ENGINE to rotate back right
    - If Angle is > 0.1 or < -0.1: DO NOT use MAIN_ENGINE. It will push you sideways!
    - FIX ANGLE FIRST, then use MAIN_ENGINE when Angle is close to 0.0.
    
    FEW-SHOT EXAMPLES:
    User: "Angle: 0.5 (Tilted Right), Vert Vel: -5.0 (Falling)"
    Assistant: execute_maneuver(action="LEFT_ENGINE", duration=3, reasoning="Angle positive, using LEFT_ENGINE to rotate counter-clockwise back to upright.")
    
    User: "Angle: -0.5 (Tilted Left), Vert Vel: -5.0 (Falling)"
    Assistant: execute_maneuver(action="RIGHT_ENGINE", duration=3, reasoning="Angle negative, using RIGHT_ENGINE to rotate clockwise back to upright.")
    
    User: "Angle: 0.0 (Upright), Vert Vel: -15.0 (Falling Fast)"
    Assistant: execute_maneuver(action="MAIN_ENGINE", duration=5, reasoning="Upright, braking descent.")
    
    User: "Angle: 0.0, Vert Vel: 5.0 (Rising)"
    Assistant: execute_maneuver(action="HOLD", duration=5, reasoning="Rising, letting gravity work.")
    
    CRITICAL: You MUST ALWAYS use the execute_maneuver tool to perform actions. 
    This is the ONLY tool available. Do NOT invent or call any other function names.
    Tool name: execute_maneuver
    Parameters: action (string), duration (integer 1-10), reasoning (string)
    
    Duration Guide:
    - Short bursts (1-3) for precision.
    - Long bursts (5-10) for major corrections.
    """
    
    return LlmAgent(
        name="commander",
        model=Gemini(
            model="gemini-2.5-flash-lite",
            retry_options=retry_config
        ),
        description="Pilot executing tactical maneuvers for lunar landing",
        instruction=commander_instruction,
        tools=[execute_maneuver]  # Function calling tool
    )
