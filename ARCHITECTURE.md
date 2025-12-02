# Project Armstrong - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     LUNAR LANDING MISSION                        │
│                  (ADK Multi-Agent System)                        │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   TELEMETRY INPUT      │
                    │  • Altitude            │
                    │  • Velocity (H/V)      │
                    │  • Angle               │
                    │  • Angular Velocity    │
                    └────────────────────────┘
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │         NAVIGATOR AGENT                        │
        │  (Strategic Advisor - LlmAgent)                │
        │  • Gemini 2.5 Flash-Lite                        │
        │  • Session Memory (InMemorySessionService)     │
        │  • Context Compaction (10 turns)               │
        │  • Role: Analyze & Recommend Strategy          │
        └────────────────────────────────────────────────┘
                                 │
                        Strategic Advice
                                 │
                                 ▼
        ┌────────────────────────────────────────────────┐
        │         COMMANDER AGENT                        │
        │  (Tactical Pilot - LlmAgent)                   │
        │  • Gemini 2.5 Flash-Lite                        │
        │  • Function Calling (execute_maneuver)         │
        │  • Role: Execute Specific Maneuvers            │
        └────────────────────────────────────────────────┘
                                 │
                    Function Call (Tool)
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  execute_maneuver      │
                    │  • MAIN_ENGINE         │
                    │  • LEFT_ENGINE         │
                    │  • RIGHT_ENGINE        │
                    │  • HOLD                │
                    │  • Duration (1-10)     │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  LUNAR LANDER ENV      │
                    │  (Gymnasium v2)        │
                    │  • Physics Simulation  │
                    │  • Reward Calculation  │
                    │  • State Updates       │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  MISSION RESULT        │
                    │  • Success/Failure     │
                    │  • Total Reward        │
                    │  • Video Recording     │
                    └────────────────────────┘
```

## ADK Components Used

### 1. **Agents** (`google.adk.agents.LlmAgent`)
- **Navigator**: Maintains conversation history via session
- **Commander**: Uses function calling for environment interaction

### 2. **Models** (`google.adk.models.google_llm.Gemini`)
- Model: `gemini-2.5-flash-lite`
- Retry configuration for API stability
- Temperature: 0.1 for deterministic decisions

### 3. **Sessions** (`google.adk.sessions.InMemorySessionService`)
- Tracks Navigator's conversation history
- Enables context-aware strategic recommendations
- Isolated per mission (session_id)

### 4. **App** (`google.adk.apps.app.App`)
- Wraps Navigator as root agent
- **EventsCompactionConfig**:
  - `compaction_interval`: 10 turns
  - `overlap_size`: 2 turns
  - Prevents unbounded chat history growth

### 5. **Runner** (`google.adk.runners.Runner`)
- Orchestrates agent execution
- Manages session lifecycle
- Streams responses asynchronously

### 6. **Tools** (Function Calling)
- `execute_maneuver`: ADK-compatible function tool
- Returns `Dict[str, Any]` with status/data structure
- Validation for action and duration parameters

## Information Flow

1. **Telemetry** → Navigator receives current state
2. **Analysis** → Navigator uses session memory to provide strategic advice
3. **Decision** → Commander combines telemetry + advice to choose maneuver
4. **Execution** → Tool call translates to environment action
5. **Feedback** → New telemetry from environment → Loop continues

## Key Features Implemented

✅ **Multi-Agent System** (Navigator + Commander)  
✅ **Custom Function Tools** (execute_maneuver with validation)  
✅ **Session Management** (InMemorySessionService for Navigator)  
✅ **Context Engineering** (EventsCompactionConfig for memory management)  
✅ **Observability** (Logging, structured responses)  

## Design Patterns

### Hierarchical Coordination
- **Navigator** = Strategic Layer (What to do?)
- **Commander** = Tactical Layer (How to do it?)
- **Tool** = Execution Layer (Do it!)

### Separation of Concerns
- Navigator maintains conversation memory
- Commander focuses on immediate execution
- Tool handles environment interface

### Error Handling
- Retry logic for API calls (5 attempts)
- Validation in execute_maneuver tool
- Fallback decisions if function calling fails

## Production Considerations

For deployment, consider:
- Replace `InMemorySessionService` with `DatabaseSessionService` for persistence
- Use `VertexAiMemoryBankService` for long-term memory across episodes
- Implement ADK evaluation framework for testing
- Add deployment artifacts (agent.yaml, .env)
