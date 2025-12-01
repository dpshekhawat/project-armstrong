import json
import sys

def generate_report(log_file="mission_log.json"):
    try:
        with open(log_file, "r") as f:
            logs = json.load(f)
    except FileNotFoundError:
        print("Log file not found. Run the mission first!")
        return

    report = "# Mission Report: Project Armstrong\n\n"
    report += "## Flight Transcript\n\n"
    
    for entry in logs:
        step = entry['step']
        telemetry = entry['telemetry']
        advice = entry['navigator_advice']
        decision = entry['commander_decision']
        result = entry['execution_result']
        
        report += f"### T-Minus {step}\n"
        report += f"**Telemetry**: {telemetry}\n\n"
        report += f"**Navigator**: *\"{advice}\"*\n\n"
        report += f"**Commander**: **{decision['action']}** ({decision['duration']} frames)\n"
        report += f"> *Reasoning: {decision['reasoning']}*\n\n"
        report += f"**Outcome**: Altitude {result['final_telemetry']['altitude']:.2f} | Reward: {result['reward_accumulated']:.2f}\n"
        report += "---\n"

    with open("mission_report.md", "w") as f:
        f.write(report)
    
    print("Report generated: mission_report.md")

if __name__ == "__main__":
    generate_report()
