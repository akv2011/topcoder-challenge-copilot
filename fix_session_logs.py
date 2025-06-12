#!/usr/bin/env python3
"""
Fix session logs structure to match validation requirements
"""

import json
from pathlib import Path

def fix_session_log_structure():
    """Convert existing session logs to expected format"""
    
    log_files = [
        "session_log_1_design_challenge__mobile_app_ui.json",
        "session_log_2_development_challenge__api_backend.json",
        "session_log_3_innovation_challenge__sustainability.json"
    ]
    
    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            print(f"⚠️ {log_file} not found")
            continue
            
        print(f"🔧 Fixing {log_file}...")
        
        try:
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            # Extract conversation from conversation_flow
            conversation = []
            final_spec = {}
            reasoning_trace = []
            
            if "conversation_flow" in data:
                for step in data["conversation_flow"]:
                    conversation.append({
                        "role": step.get("role", "unknown"),
                        "content": step.get("content", ""),
                        "timestamp": step.get("timestamp", "")
                    })
                    
                    # Extract reasoning traces
                    if "reasoning_trace" in step:
                        for trace in step["reasoning_trace"]:
                            reasoning_trace.append({
                                "step": step.get("step"),
                                "reasoning": trace,
                                "confidence": 0.8  # Default confidence
                            })
            
            # Extract final spec from the data
            if "final_challenge_spec" in data:
                final_spec = data["final_challenge_spec"]
            else:
                # Create basic spec from available data
                final_spec = {
                    "title": data.get("challenge_type", "Challenge"),
                    "platform": data.get("platform", "Topcoder"),
                    "challenge_type": data.get("challenge_type", "Design"),
                    "user_goal": data.get("user_goal", ""),
                    "scope": data.get("agreed_scope", {}),
                    "requirements": data.get("requirements", []),
                    "timeline": data.get("timeline", {}),
                    "deliverables": data.get("deliverables", [])
                }
            
            # Create new structure
            fixed_data = {
                "session_id": data.get("session_id", "session_001"),
                "timestamp": data.get("timestamp", "2025-06-10T04:00:00Z"),
                "conversation": conversation,
                "final_spec": final_spec,
                "reasoning_trace": reasoning_trace,
                "metadata": {
                    "platform": data.get("platform", "Topcoder"),
                    "challenge_type": data.get("challenge_type", "Design"),
                    "user_goal": data.get("user_goal", ""),
                    "status": "completed"
                }
            }
            
            # Write fixed data
            with open(log_path, 'w') as f:
                json.dump(fixed_data, f, indent=2)
                
            print(f"✅ Fixed {log_file}")
            print(f"   - Conversation: {len(conversation)} messages")
            print(f"   - Final spec: {len(final_spec)} fields") 
            print(f"   - Reasoning trace: {len(reasoning_trace)} entries")
            
        except Exception as e:
            print(f"❌ Error fixing {log_file}: {e}")

if __name__ == "__main__":
    fix_session_log_structure()
