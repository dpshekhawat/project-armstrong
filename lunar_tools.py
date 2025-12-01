import gymnasium as gym
import numpy as np
import json
import imageio
from typing import Dict, Any

class LunarLanderInterface:
    def __init__(self):
        self.env = gym.make('LunarLander-v2', render_mode="rgb_array")
        self.current_state, _ = self.env.reset()
        self.total_reward = 0
        self.steps = 0
        self.done = False
        self.frames = [] # Store frames for video

    def reset(self):
        self.current_state, _ = self.env.reset()
        self.total_reward = 0
        self.steps = 0
        self.done = False
        self.frames = []
        return self.get_telemetry()

    def get_telemetry(self):
        """
        Extracts readable telemetry from the state vector.
        State vector: [x, y, vx, vy, angle, angular_vel, leg1, leg2]
        """
        s = self.current_state
        
        # Interpret values for the LLM
        # x: 0 is center. -1 is left, 1 is right.
        # y: 0 is landing pad (approx), starts at ~1.4
        # angle: 0 is upright.
        
        telemetry = {
            "altitude": float(s[1]),
            "horizontal_position": float(s[0]),
            "vertical_velocity": float(s[3]),
            "horizontal_velocity": float(s[2]),
            "angle": float(s[4]),
            "angular_velocity": float(s[5]),
            "left_leg_contact": bool(s[6]),
            "right_leg_contact": bool(s[7]),
            "steps_taken": self.steps
        }
        return telemetry

    def get_telemetry_description(self):
        """
        Returns a natural language description of the state for the Navigator agent.
        """
        t = self.get_telemetry()
        
        desc = f"Altitude: {t['altitude']:.2f}. "
        desc += f"Position X: {t['horizontal_position']:.2f} (0 is center). "
        desc += f"Vertical Velocity: {t['vertical_velocity']:.2f} (Negative is falling). "
        desc += f"Horizontal Velocity: {t['horizontal_velocity']:.2f}. "
        desc += f"Angle: {t['angle']:.2f} radians. "
        
        status = "Flying"
        if t['left_leg_contact'] or t['right_leg_contact']:
            status = "Touchdown imminent/Landed"
            
        desc += f"Status: {status}."
        return desc

    def execute_maneuver(self, action_name, duration=1):
        """
        Executes an action for a specific number of frames.
        Actions: 0: Do nothing, 1: Fire right engine (push left), 2: Main engine, 3: Left engine (push right)
        """
        action_map = {
            "HOLD": 0,
            "MAIN_ENGINE": 2,
            "LEFT_ENGINE": 3, # Fires left engine, pushes lander to the RIGHT
            "RIGHT_ENGINE": 1 # Fires right engine, pushes lander to the LEFT
        }
        
        action_code = action_map.get(action_name, 0)
        
        step_rewards = 0
        info_log = []
        
        for _ in range(duration):
            if self.done:
                break
                
            next_state, reward, terminated, truncated, info = self.env.step(action_code)
            self.done = terminated or truncated
            self.current_state = next_state
            self.total_reward += reward
            self.steps += 1
            step_rewards += reward
            
            # Capture frame
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)

        return {
            "final_telemetry": self.get_telemetry(),
            "reward_accumulated": step_rewards,
            "done": self.done,
            "action_executed": action_name,
            "duration": duration
        }

    def close(self):
        self.env.close()

    def save_video(self, filename="mission_replay.mp4", fps=30):
        """
        Saves the recorded frames to a video file.
        """
        if not self.frames:
            print("No frames to save.")
            return
        
        try:
            imageio.mimsave(filename, self.frames, fps=fps)
            print(f"Video saved to {filename}")
        except Exception as e:
            print(f"Error saving video: {e}")
