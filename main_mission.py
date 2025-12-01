import time
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any
from lunar_tools import LunarLanderInterface
from agents import create_navigator_agent, create_commander_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, EventsCompactionConfig
from google.genai import types
import asyncio

async def main():
    # 1. Setup
    load_dotenv() # Load environment variables from .env file
    print("--- PROJECT ARMSTRONG: MISSION CONTROL ---")
    
    # Check for API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Please set your GOOGLE_API_KEY environment variable.")
        print("Example: $env:GOOGLE_API_KEY='your_key_here'")
        return {"status": "ABORTED", "error": "No API key"}
    
    # Initialize Components
    lander = LunarLanderInterface()
    
    # Create ADK agents
    navigator = create_navigator_agent(api_key=api_key)
    commander = create_commander_agent(api_key=api_key)
    
    # Create App with Context Compaction for Navigator's session memory
    # This prevents unbounded chat history growth
    app = App(
        name="lunar_landing_mission",
        root_agent=navigator,  # Navigator is the entry point
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=10,  # Compact after every 10 conversation turns
            overlap_size=2  # Keep last 2 turns for context
        )
    )
    
    # Create Session Service and Runners
    # CRITICAL: Both agents share the same session service to avoid creating new sessions
    session_service = InMemorySessionService()
    navigator_runner = Runner(
        app=app,
        session_service=session_service
    )
    
    # Create separate runner for Commander (same session service)
    commander_runner = Runner(
        agent=commander,
        app_name="lunar_landing_mission",
        session_service=session_service
    )
    
    # Create session for this mission
    APP_NAME = "lunar_landing_mission"
    USER_ID = "mission_control"
    SESSION_ID = "flight_001"
    
    mission_log = []
    episode_reward = 0
    
    print("Mission Start. Initializing systems...")
    time.sleep(1)
    
    # Create sessions for both agents
    # Navigator session for strategic advice
    try:
        nav_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    except:
        nav_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
    
    # Commander session for tactical decisions
    COMMANDER_SESSION_ID = f"{SESSION_ID}_commander"
    try:
        cmd_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=COMMANDER_SESSION_ID
        )
    except:
        cmd_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=COMMANDER_SESSION_ID
        )
    
    # 2. Mission Loop
    max_steps = 100 # Safety limit for the demo
    step_count = 0
    
    try:
        while not lander.done and step_count < max_steps:
            step_count += 1
            print(f"\n--- T-Minus {step_count} ---")
            
            # A. Get Telemetry
            telemetry_desc = lander.get_telemetry_description()
            print(f"[TELEMETRY] {telemetry_desc}")
            
            # B. Navigator Analysis (using ADK Runner with session)
            print("[NAVIGATOR] Analyzing...")
            navigator_query = types.Content(
                role="user",
                parts=[types.Part(text=telemetry_desc)]
            )
            
            advice = ""
            async for event in navigator_runner.run_async(
                user_id=USER_ID,
                session_id=nav_session.id,
                new_message=navigator_query
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    advice = event.content.parts[0].text
                    
            print(f"[NAVIGATOR] Advice: {advice}")
            
            # C. Commander Decision (using shared session pattern)
            print("[COMMANDER] Deciding maneuver...")
            commander_prompt = types.Content(
                role="user",
                parts=[types.Part(text=f"""
            Current Telemetry: {telemetry_desc}
            Navigator Advice: {advice}
            
            Determine the best maneuver and execute it using the execute_maneuver tool.
            """)]
            )
            
            # Use the same Runner pattern with timeout
            decision = {"action": "HOLD", "duration": 1, "reasoning": "Fallback"}
            try:
                # Run Commander with timeout using run_async
                async def run_commander_with_timeout():
                    events_list = []
                    async for event in commander_runner.run_async(
                        user_id=USER_ID,
                        session_id=cmd_session.id,
                        new_message=commander_prompt
                    ):
                        events_list.append(event)
                    return events_list
                
                events = await asyncio.wait_for(
                    run_commander_with_timeout(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                print("[WARNING] Commander timeout - using fallback HOLD")
                events = []
            
            # Extract decision from FIRST function call found
            found = False
            for event in events:
                if found:
                    break
                if hasattr(event, 'content') and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fn = part.function_call
                            decision = {
                                "action": fn.args.get("action", "HOLD"),
                                "duration": int(fn.args.get("duration", 1)),
                                "reasoning": fn.args.get("reasoning", "No reasoning provided")
                            }
                            found = True
                            break
                        
            print(f"[COMMANDER] Action: {decision['action']} for {decision['duration']} frames.")
            print(f"[COMMANDER] Reasoning: {decision['reasoning']}")
            
            # D. Execution
            result = lander.execute_maneuver(decision['action'], decision['duration'])
            episode_reward += result['reward_accumulated']
            
            # E. Logging
            log_entry = {
                "step": step_count,
                "telemetry": telemetry_desc,
                "navigator_advice": advice,
                "commander_decision": decision,
                "execution_result": result
            }
            mission_log.append(log_entry)
            
            # Rate Limit Protection:
            # Gemini 2.5 Flash-Lite Free Tier allows 15 RPM.
            # We make 2 requests per step (Navigator + Commander).
            # So we can do max 7.5 steps per minute = 1 step every 8 seconds.
            # We sleep for 9 seconds to be safe.
            await asyncio.sleep(9.0) 
            
        # 3. Mission End
        print("\n--- MISSION END ---")
        print(f"Total Reward: {episode_reward}")
        
        result_status = "INCOMPLETE"
        if lander.total_reward >= 200:
            print("RESULT: SUCCESSFUL LANDING")
            result_status = "SUCCESS"
        elif lander.total_reward <= -100:
            print("RESULT: CRASH")
            result_status = "CRASH"
        else:
            print("RESULT: INCOMPLETE / HOVERING")
            
        return {
            "status": result_status,
            "total_reward": episode_reward,
            "steps": step_count,
            "log": mission_log
        }
            
    except KeyboardInterrupt:
        print("\nMission Aborted by User.")
        return {"status": "ABORTED", "total_reward": episode_reward, "steps": step_count}
    finally:
        # Save Logs
        with open("mission_log.json", "w") as f:
            json.dump(mission_log, f, indent=2)
        print("Mission logs saved to 'mission_log.json'.")
        
        # Save Video
        print("Saving mission replay...")
        lander.save_video("mission_replay.mp4")
        
        lander.close()

if __name__ == "__main__":
    asyncio.run(main())
