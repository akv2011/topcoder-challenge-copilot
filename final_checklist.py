#!/usr/bin/env python3
"""
AI Copilot Agent - Final Deliverables Checklist
This script provides a comprehensive checklist of all deliverables for submission.
"""

import os
import json
from pathlib import Path
from datetime import datetime

class DeliverablesChecklist:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.checklist = {}
        
    def check_core_deliverables(self):
        """Check core required deliverables"""
        print("📋 CORE DELIVERABLES CHECKLIST")
        print("="*50)
        
        core_items = {
            "1. Agent Backend Implementation": {
                "file": "backend/main.py",
                "requirements": [
                    "Scoping dialogue logic",
                    "Schema-aware Q&A logic", 
                    "RAG integration",
                    "Reasoning trace generation"
                ]
            },
            "2. Session Logs (3 required)": {
                "files": [
                    "session_log_1_design_challenge__mobile_app_ui.json",
                    "session_log_2_development_challenge__api_backend.json",
                    "session_log_3_innovation_challenge__sustainability.json"
                ]
            },
            "3. Documentation": {
                "file": "README.md",
                "requirements": [
                    "Agent orchestration setup",
                    "Memory handling",
                    "Prompt strategies", 
                    "Schema configurability"
                ]
            },
            "4. Dependencies": {
                "files": [
                    "backend/requirements.txt",
                    "frontend/package.json"
                ]
            }
        }
        
        for item, details in core_items.items():
            print(f"\n{item}")
            print("-" * 30)
            
            if "file" in details:
                file_path = self.root_path / details["file"]
                status = "✅" if file_path.exists() else "❌"
                print(f"  {status} File: {details['file']}")
                
                if file_path.exists() and "requirements" in details:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                    
                    for req in details["requirements"]:
                        req_found = any(word in content for word in req.lower().split())
                        req_status = "✅" if req_found else "❌"
                        print(f"    {req_status} {req}")
            
            if "files" in details:
                for file_name in details["files"]:
                    file_path = self.root_path / file_name
                    status = "✅" if file_path.exists() else "❌"
                    print(f"  {status} {file_name}")
    
    def check_technical_requirements(self):
        """Check technical implementation requirements"""
        print(f"\n📋 TECHNICAL REQUIREMENTS CHECKLIST")
        print("="*50)
        
        tech_requirements = {
            "Agent Framework": {
                "required": "LangGraph or CrewAI",
                "check_for": ["langgraph", "crewai", "StateGraph"]
            },
            "LLM Integration": {
                "required": "OpenAI GPT-4",
                "check_for": ["openai", "gpt-4", "ChatOpenAI"]
            },
            "RAG Framework": {
                "required": "LangChain or LlamaIndex",
                "check_for": ["langchain", "llamaindex", "vector", "embedding"]
            },
            "Vector Store": {
                "required": "Qdrant, Pinecone, or FAISS",
                "check_for": ["qdrant", "pinecone", "faiss", "chroma"]
            },
            "Server Framework": {
                "required": "FastAPI or Express",
                "check_for": ["fastapi", "express", "uvicorn"]
            },
            "Memory Management": {
                "required": "Session state persistence",
                "check_for": ["session", "memory", "state", "history"]
            },
            "Dynamic Schema": {
                "required": "Configurable challenge fields",
                "check_for": ["schema", "dynamic", "platform", "field"]
            },
            "Feedback Loop": {
                "required": "User feedback handling",
                "check_for": ["feedback", "adapt", "reject", "accept"]
            }
        }
        
        backend_file = self.root_path / "backend" / "main.py"
        if backend_file.exists():
            with open(backend_file, 'r') as f:
                content = f.read().lower()
                
            for req_name, req_info in tech_requirements.items():
                found = any(keyword in content for keyword in req_info["check_for"])
                status = "✅" if found else "❌"
                print(f"{status} {req_name}: {req_info['required']}")
        else:
            print("❌ Backend file not found - cannot check technical requirements")
    
    def check_bonus_features(self):
        """Check bonus features implementation"""
        print(f"\n📋 BONUS FEATURES CHECKLIST")
        print("="*50)
        
        bonus_features = {
            "Vision Model Support": {
                "description": "Upload mockups for context",
                "check_files": ["backend/main.py"],
                "check_for": ["vision", "image", "upload", "gpt-4-vision", "multimodal"]
            },
            "React Frontend Integration": {
                "description": "Chat interface, Checklist view, Challenge spec viewer",
                "check_files": ["frontend/src/App.tsx"],
                "check_for": ["chat", "checklist", "spec", "interface"]
            }
        }
        
        for feature_name, feature_info in bonus_features.items():
            print(f"\n{feature_name}")
            print("-" * 30)
            print(f"Description: {feature_info['description']}")
            
            found = False
            for file_path in feature_info["check_files"]:
                full_path = self.root_path / file_path
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        content = f.read().lower()
                    
                    if any(keyword in content for keyword in feature_info["check_for"]):
                        found = True
                        break
            
            status = "✅ IMPLEMENTED" if found else "❌ NOT IMPLEMENTED"
            print(f"Status: {status}")
    
    def check_output_format(self):
        """Check if output format meets specifications"""
        print(f"\n📋 OUTPUT FORMAT CHECKLIST")
        print("="*50)
        
        required_output_fields = [
            "Structured challenge fields (JSON)",
            "Reasoning per field",
            "Confidence level (optional)"
        ]
        
        # Check session logs for proper output format
        log_files = [
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json",
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        for log_file in log_files:
            log_path = self.root_path / log_file
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        log_data = json.load(f)
                    
                    final_spec = log_data.get("final_spec", {})
                    reasoning_trace = log_data.get("reasoning_trace", [])
                    
                    print(f"\n{log_file}:")
                    print(f"  ✅ Structured fields: {len(final_spec)} fields")
                    print(f"  ✅ Reasoning trace: {len(reasoning_trace)} entries")
                    
                    # Check for confidence levels
                    has_confidence = any("confidence" in entry for entry in reasoning_trace if isinstance(entry, dict))
                    conf_status = "✅" if has_confidence else "⚠️"
                    print(f"  {conf_status} Confidence levels: {'Present' if has_confidence else 'Optional - not required'}")
                    
                except Exception as e:
                    print(f"  ❌ {log_file}: Error reading - {e}")
            else:
                print(f"  ❌ {log_file}: Missing")
    
    def check_documentation_quality(self):
        """Check documentation completeness"""
        print(f"\n📋 DOCUMENTATION QUALITY CHECKLIST")
        print("="*50)
        
        readme_path = self.root_path / "README.md"
        if not readme_path.exists():
            print("❌ README.md file missing")
            return
            
        with open(readme_path, 'r') as f:
            content = f.read().lower()
            
        required_sections = {
            "Agent orchestration setup": ["agent", "setup", "orchestration"],
            "Memory handling": ["memory", "state", "session"],
            "Prompt strategies": ["prompt", "strategy", "question"],
            "Schema configurability": ["schema", "config", "platform"],
            "Installation instructions": ["install", "setup", "requirements"],
            "Usage examples": ["usage", "example", "how to"],
            "Tech stack details": ["tech", "stack", "dependencies"]
        }
        
        for section, keywords in required_sections.items():
            found = any(keyword in content for keyword in keywords)
            status = "✅" if found else "❌"
            print(f"{status} {section}")
    
    def generate_submission_summary(self):
        """Generate final submission summary"""
        print(f"\n🏆 SUBMISSION SUMMARY")
        print("="*50)
        
        # Count files
        file_counts = {
            "Backend files": len(list((self.root_path / "backend").glob("*.py"))),
            "Frontend files": len(list((self.root_path / "frontend" / "src").glob("*.tsx"))) if (self.root_path / "frontend" / "src").exists() else 0,
            "Session logs": len([f for f in ["session_log_1_design_challenge__mobile_app_ui.json", 
                                           "session_log_2_development_challenge__api_backend.json",
                                           "session_log_3_innovation_challenge__sustainability.json"] 
                                if (self.root_path / f).exists()]),
            "Config files": len(list(self.root_path.glob("*.json"))) + len(list(self.root_path.glob("*.txt")))
        }
        
        print("📊 File Summary:")
        for category, count in file_counts.items():
            print(f"  • {category}: {count}")
        
        # Check completeness
        critical_files = [
            "backend/main.py",
            "README.md",
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json", 
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        missing_critical = []
        for file_path in critical_files:
            if not (self.root_path / file_path).exists():
                missing_critical.append(file_path)
        
        print(f"\n🎯 Readiness Assessment:")
        if not missing_critical:
            print("✅ ALL CRITICAL DELIVERABLES PRESENT")
            print("🚀 READY FOR SUBMISSION!")
        else:
            print("❌ MISSING CRITICAL FILES:")
            for missing in missing_critical:
                print(f"  • {missing}")
        
        # Generate submission timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n📅 Validation completed: {timestamp}")
        
        return {
            "timestamp": timestamp,
            "file_counts": file_counts,
            "missing_critical": missing_critical,
            "ready_for_submission": len(missing_critical) == 0
        }
    
    def run_full_checklist(self):
        """Run complete deliverables checklist"""
        print("🎯 AI COPILOT AGENT - FINAL DELIVERABLES CHECKLIST")
        print("="*60)
        print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        self.check_core_deliverables()
        self.check_technical_requirements()
        self.check_bonus_features()
        self.check_output_format()
        self.check_documentation_quality()
        summary = self.generate_submission_summary()
        
        # Save checklist results
        checklist_path = self.root_path / "deliverables_checklist.json"
        with open(checklist_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Checklist results saved to: {checklist_path}")
        
        return summary

def main():
    """Main function"""
    root_path = os.path.dirname(os.path.abspath(__file__))
    checker = DeliverablesChecklist(root_path)
    
    try:
        summary = checker.run_full_checklist()
        
        # Return exit code based on readiness
        return 0 if summary['ready_for_submission'] else 1
        
    except Exception as e:
        print(f"❌ Checklist error: {e}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
