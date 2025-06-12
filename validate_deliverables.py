#!/usr/bin/env python3
"""
AI Copilot Agent - Deliverables Validation Script
This script validates all required functionality and deliverables for the project.
"""

import os
import json
import sys
from pathlib import Path
import importlib.util
import ast

class DeliverableValidator:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.results = {}
        self.errors = []
        self.warnings = []
        
    def log_result(self, category, item, status, details=""):
        if category not in self.results:
            self.results[category] = {}
        self.results[category][item] = {
            "status": status,
            "details": details
        }
        
    def validate_file_structure(self):
        """Validate required file structure"""
        print("🔍 Validating File Structure...")
        
        required_files = [
            "backend/main.py",
            "backend/requirements.txt", 
            "frontend/src/App.tsx",
            "frontend/package.json",
            "README.md",
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json", 
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        for file_path in required_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                self.log_result("file_structure", file_path, "✅ PASS", f"File exists: {full_path}")
            else:
                self.log_result("file_structure", file_path, "❌ FAIL", f"Missing file: {full_path}")
                self.errors.append(f"Missing required file: {file_path}")
    
    def validate_backend_functionality(self):
        """Validate backend implementation"""
        print("🔍 Validating Backend Functionality...")
        
        main_py_path = self.root_path / "backend" / "main.py"
        if not main_py_path.exists():
            self.log_result("backend", "main.py", "❌ FAIL", "File not found")
            return
            
        try:
            with open(main_py_path, 'r') as f:
                content = f.read()
                
            # Check for required components
            required_components = {
                "FastAPI": "FastAPI" in content,
                "LangGraph": "langgraph" in content.lower() or "StateGraph" in content,
                "RAG_Integration": "rag" in content.lower() or "vector" in content.lower(),
                "Dynamic_Schema": "schema" in content.lower() and "dynamic" in content.lower(),
                "Session_Management": "session" in content.lower(),
                "Reasoning_Trace": "reasoning" in content.lower() or "trace" in content.lower(),
                "Feedback_Loop": "feedback" in content.lower(),
                "Vision_Support": "vision" in content.lower() or "image" in content.lower(),
                "CORS_Middleware": "CORS" in content or "CORSMiddleware" in content
            }
            
            for component, exists in required_components.items():
                status = "✅ PASS" if exists else "❌ FAIL"
                self.log_result("backend", component, status, f"Found in code: {exists}")
                if not exists:
                    self.errors.append(f"Backend missing: {component}")
                    
            # Check API endpoints
            required_endpoints = [
                "/chat",
                "/sessions",
                "/schemas",
                "/upload"
            ]
            
            for endpoint in required_endpoints:
                if endpoint in content:
                    self.log_result("backend_endpoints", endpoint, "✅ PASS", "Endpoint defined")
                else:
                    self.log_result("backend_endpoints", endpoint, "❌ FAIL", "Endpoint missing")
                    self.errors.append(f"Missing API endpoint: {endpoint}")
                    
        except Exception as e:
            self.log_result("backend", "validation", "❌ FAIL", f"Error reading file: {e}")
            self.errors.append(f"Backend validation error: {e}")
    
    def validate_frontend_functionality(self):
        """Validate frontend implementation"""
        print("🔍 Validating Frontend Functionality...")
        
        app_tsx_path = self.root_path / "frontend" / "src" / "App.tsx"
        if not app_tsx_path.exists():
            self.log_result("frontend", "App.tsx", "❌ FAIL", "File not found")
            return
            
        try:
            with open(app_tsx_path, 'r') as f:
                content = f.read()
                
            # Check for required React components
            required_components = {
                "Chat_Interface": "chat" in content.lower(),
                "Checklist_View": "checklist" in content.lower() or "progress" in content.lower(),
                "Challenge_Spec_Viewer": "spec" in content.lower() or "challenge" in content.lower(),
                "File_Upload": "upload" in content.lower() or "file" in content.lower(),
                "React_Hooks": "useState" in content or "useEffect" in content,
                "TypeScript": "interface" in content or "type" in content
            }
            
            for component, exists in required_components.items():
                status = "✅ PASS" if exists else "❌ FAIL"
                self.log_result("frontend", component, status, f"Found in code: {exists}")
                if not exists:
                    self.errors.append(f"Frontend missing: {component}")
                    
        except Exception as e:
            self.log_result("frontend", "validation", "❌ FAIL", f"Error reading file: {e}")
            self.errors.append(f"Frontend validation error: {e}")
    
    def validate_session_logs(self):
        """Validate session logs"""
        print("🔍 Validating Session Logs...")
        
        log_files = [
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json",
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        for log_file in log_files:
            log_path = self.root_path / log_file
            if not log_path.exists():
                self.log_result("session_logs", log_file, "❌ FAIL", "File not found")
                self.errors.append(f"Missing session log: {log_file}")
                continue
                
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    
                # Validate log structure
                required_fields = [
                    "session_id",
                    "conversation",
                    "final_spec",
                    "reasoning_trace"
                ]
                
                missing_fields = [field for field in required_fields if field not in log_data]
                
                if missing_fields:
                    self.log_result("session_logs", log_file, "❌ FAIL", f"Missing fields: {missing_fields}")
                    self.errors.append(f"Session log {log_file} missing fields: {missing_fields}")
                else:
                    # Check conversation depth
                    conversation_length = len(log_data.get("conversation", []))
                    if conversation_length >= 5:
                        self.log_result("session_logs", log_file, "✅ PASS", f"Valid session with {conversation_length} turns")
                    else:
                        self.log_result("session_logs", log_file, "⚠️ WARNING", f"Short conversation: {conversation_length} turns")
                        self.warnings.append(f"Session log {log_file} has short conversation")
                        
            except json.JSONDecodeError as e:
                self.log_result("session_logs", log_file, "❌ FAIL", f"Invalid JSON: {e}")
                self.errors.append(f"Session log {log_file} has invalid JSON: {e}")
            except Exception as e:
                self.log_result("session_logs", log_file, "❌ FAIL", f"Error reading file: {e}")
                self.errors.append(f"Session log {log_file} error: {e}")
    
    def validate_requirements(self):
        """Validate requirements and dependencies"""
        print("🔍 Validating Requirements...")
        
        # Backend requirements
        backend_req_path = self.root_path / "backend" / "requirements.txt"
        if backend_req_path.exists():
            try:
                with open(backend_req_path, 'r') as f:
                    requirements = f.read().lower()
                    
                required_packages = {
                    "fastapi": "fastapi" in requirements,
                    "uvicorn": "uvicorn" in requirements,
                    "openai": "openai" in requirements,
                    "langgraph": "langgraph" in requirements,
                    "langchain": "langchain" in requirements,
                    "pydantic": "pydantic" in requirements,
                    "python-multipart": "python-multipart" in requirements or "multipart" in requirements
                }
                
                for package, exists in required_packages.items():
                    status = "✅ PASS" if exists else "❌ FAIL"
                    self.log_result("backend_requirements", package, status, f"Found: {exists}")
                    if not exists:
                        self.errors.append(f"Missing Python package: {package}")
                        
            except Exception as e:
                self.log_result("backend_requirements", "validation", "❌ FAIL", f"Error: {e}")
        else:
            self.log_result("backend_requirements", "file", "❌ FAIL", "requirements.txt not found")
            
        # Frontend package.json
        frontend_pkg_path = self.root_path / "frontend" / "package.json"
        if frontend_pkg_path.exists():
            try:
                with open(frontend_pkg_path, 'r') as f:
                    pkg_data = json.load(f)
                    
                dependencies = pkg_data.get("dependencies", {})
                dev_dependencies = pkg_data.get("devDependencies", {})
                all_deps = {**dependencies, **dev_dependencies}
                
                required_packages = {
                    "react": "react" in all_deps,
                    "typescript": "typescript" in all_deps,
                    "vite": "vite" in all_deps,
                    "tailwindcss": "tailwindcss" in all_deps
                }
                
                for package, exists in required_packages.items():
                    status = "✅ PASS" if exists else "❌ FAIL"
                    self.log_result("frontend_requirements", package, status, f"Found: {exists}")
                    if not exists:
                        self.errors.append(f"Missing Node package: {package}")
                        
            except Exception as e:
                self.log_result("frontend_requirements", "validation", "❌ FAIL", f"Error: {e}")
        else:
            self.log_result("frontend_requirements", "file", "❌ FAIL", "package.json not found")
    
    def validate_readme(self):
        """Validate README documentation"""
        print("🔍 Validating README...")
        
        readme_path = self.root_path / "README.md"
        if not readme_path.exists():
            self.log_result("documentation", "README.md", "❌ FAIL", "File not found")
            self.errors.append("Missing README.md")
            return
            
        try:
            with open(readme_path, 'r') as f:
                content = f.read().lower()
                
            required_sections = {
                "agent_orchestration": "agent" in content and "orchestration" in content,
                "memory_handling": "memory" in content,
                "prompt_strategies": "prompt" in content and "strateg" in content,
                "schema_configurability": "schema" in content and "config" in content,
                "setup_instructions": "setup" in content or "install" in content,
                "tech_stack": "tech" in content or "stack" in content or "dependencies" in content
            }
            
            for section, exists in required_sections.items():
                status = "✅ PASS" if exists else "❌ FAIL"
                self.log_result("documentation", section, status, f"Found: {exists}")
                if not exists:
                    self.warnings.append(f"README missing section: {section}")
                    
        except Exception as e:
            self.log_result("documentation", "validation", "❌ FAIL", f"Error: {e}")
    
    def validate_bonus_features(self):
        """Validate bonus features"""
        print("🔍 Validating Bonus Features...")
        
        # Check for vision model support
        backend_path = self.root_path / "backend" / "main.py"
        if backend_path.exists():
            with open(backend_path, 'r') as f:
                backend_content = f.read().lower()
                
            vision_support = any(keyword in backend_content for keyword in [
                "vision", "image", "upload", "file", "gpt-4-vision", "multimodal"
            ])
            
            status = "✅ PASS" if vision_support else "❌ FAIL"
            self.log_result("bonus_features", "vision_model_support", status, f"Found: {vision_support}")
            
        # Check for React frontend integration
        frontend_path = self.root_path / "frontend" / "src" / "App.tsx"
        if frontend_path.exists():
            self.log_result("bonus_features", "react_frontend", "✅ PASS", "React frontend implemented")
        else:
            self.log_result("bonus_features", "react_frontend", "❌ FAIL", "React frontend missing")
    
    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "="*60)
        print("🏆 DELIVERABLES VALIDATION REPORT")
        print("="*60)
        
        total_checks = 0
        passed_checks = 0
        
        for category, items in self.results.items():
            print(f"\n📋 {category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            for item, result in items.items():
                print(f"  {result['status']} {item}: {result['details']}")
                total_checks += 1
                if result['status'] == "✅ PASS":
                    passed_checks += 1
        
        print(f"\n📊 SUMMARY")
        print("-" * 40)
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        print(f"Success Rate: {(passed_checks/total_checks*100):.1f}%")
        
        if self.errors:
            print(f"\n❌ CRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"\n⚠️ WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("-" * 40)
        
        if len(self.errors) == 0:
            print("✅ ALL DELIVERABLES READY FOR SUBMISSION!")
        elif len(self.errors) <= 3:
            print("⚠️ MOSTLY READY - FEW MINOR ISSUES TO FIX")
        else:
            print("❌ SIGNIFICANT ISSUES NEED ATTENTION")
            
        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "results": self.results
        }
    
    def run_validation(self):
        """Run all validation checks"""
        print("🚀 Starting AI Copilot Agent Deliverables Validation")
        print("="*60)
        
        self.validate_file_structure()
        self.validate_backend_functionality() 
        self.validate_frontend_functionality()
        self.validate_session_logs()
        self.validate_requirements()
        self.validate_readme()
        self.validate_bonus_features()
        
        return self.generate_report()

def main():
    """Main validation function"""
    root_path = os.path.dirname(os.path.abspath(__file__))
    validator = DeliverableValidator(root_path)
    
    try:
        report = validator.run_validation()
        
        # Save report to file
        report_path = os.path.join(root_path, "validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_path}")
        
        # Return exit code based on results
        if len(report['errors']) == 0:
            return 0  # Success
        elif len(report['errors']) <= 3:
            return 1  # Minor issues
        else:
            return 2  # Major issues
            
    except Exception as e:
        print(f"❌ Validation script error: {e}")
        return 3  # Script error

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
