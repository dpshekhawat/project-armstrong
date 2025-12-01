import time
import json
import os
import numpy as np
import asyncio
from dotenv import load_dotenv
from lunar_tools import LunarLanderInterface
from agents import create_navigator_agent, create_commander_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App, EventsCompactionConfig
from google.genai import types

async def run_episode(episode_id, api_key):
    print(f"\n--- STARTING EPISODE {episode_id} ---")
    
    # Initialize Components
    lander = LunarLanderInterface()
    navigator = create_navigator_agent(api_key=api_key)
    commander = create_commander_agent(api_key=api_key)
    
    # Create App with Context Compaction
    app = App(
        name="lunar_eval",
        root_agent=navigator,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=10,
            overlap_size=2
        )
    )
    
    session_service = InMemorySessionService()
    navigator_runner = Runner(app=app, session_service=session_service)
    
    # Create separate runner for Commander (same session service)
    commander_runner = Runner(
        agent=commander,
        app_name="lunar_eval",
        session_service=session_service
    )
    
    # Create session
    APP_NAME = "lunar_eval"
    USER_ID = "evaluator"
    SESSION_ID = f"eval_episode_{episode_id}"
    
    # Create sessions for both agents
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
    
    # Commander session
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
    
    episode_reward = 0
    step_count = 0
    max_steps = 80 # Limit steps to prevent infinite loops
    
    mission_log = []
    
    try:
        while not lander.done and step_count < max_steps:
            step_count += 1
            
            # A. Get Telemetry
            telemetry_desc = lander.get_telemetry_description()
            
            # B. Navigator Analysis (using ADK Runner)
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
            
            # C. Commander Decision (using shared session pattern)
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
            
            print(f"Ep {episode_id} | Step {step_count} | Reward: {episode_reward:.2f} | Action: {decision['action']}")
            
            # Rate Limit Protection (Safe 9s)
            await asyncio.sleep(9.0)
            
    except Exception as e:
        print(f"Episode {episode_id} Error: {e}")
    finally:
        lander.close()
        
    return {
        "episode_id": episode_id,
        "total_reward": float(episode_reward),
        "steps": step_count,
        "success": bool(episode_reward >= 200),
        "log": mission_log
    }

async def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found.")
        return

    NUM_EPISODES = 3
    results = []
    
    print(f"Starting Evaluation of {NUM_EPISODES} episodes...")
    
    for i in range(1, NUM_EPISODES + 1):
        result = await run_episode(i, api_key)
        results.append(result)
        print(f"Episode {i} Finished. Reward: {result['total_reward']:.2f}, Success: {result['success']}")
        
    # Calculate Metrics
    rewards = [r["total_reward"] for r in results]
    successes = [r["success"] for r in results]
    
    avg_reward = float(np.mean(rewards))
    success_rate = float(np.mean(successes) * 100)
    
    print("\n--- EVALUATION REPORT ---")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    report = {
        "average_reward": avg_reward,
        "success_rate": success_rate,
        "episodes": results
    }
    
    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Detailed report saved to 'evaluation_report.json'")

if __name__ == "__main__":
    asyncio.run(main())
