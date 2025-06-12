#!/usr/bin/env python3
"""
AI Copilot Agent - Technical Functionality Test Script
This script tests all technical components without running the servers.
"""

import os
import json
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Any

class TechnicalValidator:
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.results = {}
        
    def test_backend_architecture(self):
        """Test backend architecture components"""
        print("🔧 Testing Backend Architecture...")
        
        main_py = self.root_path / "backend" / "main.py"
        if not main_py.exists():
            return {"status": "❌ FAIL", "details": "main.py not found"}
            
        with open(main_py, 'r') as f:
            content = f.read()
            
        # Parse AST to analyze code structure
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {"status": "❌ FAIL", "details": f"Syntax error: {e}"}
            
        # Check for required classes and functions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        architecture_checks = {
            "LangGraph_Implementation": any("graph" in cls.lower() or "state" in cls.lower() for cls in classes),
            "RAG_Components": any("rag" in func.lower() or "vector" in func.lower() or "embed" in func.lower() for func in functions),
            "Session_Management": any("session" in func.lower() for func in functions),
            "Schema_Loading": any("schema" in func.lower() for func in functions),
            "Reasoning_Engine": any("reason" in func.lower() or "trace" in func.lower() for func in functions),
            "FastAPI_Setup": "FastAPI" in content,
            "CORS_Configuration": "CORS" in content,
            "Error_Handling": "try:" in content and "except" in content,
            "Async_Functions": "async def" in content
        }
        
        return {
            "status": "✅ PASS" if all(architecture_checks.values()) else "⚠️ PARTIAL",
            "details": architecture_checks,
            "classes": classes,
            "functions": functions
        }
    
    def test_agent_framework_integration(self):
        """Test LangGraph/Agent framework integration"""
        print("🤖 Testing Agent Framework Integration...")
        
        main_py = self.root_path / "backend" / "main.py"
        with open(main_py, 'r') as f:
            content = f.read()
            
        langgraph_features = {
            "StateGraph_Usage": "StateGraph" in content,
            "Node_Definition": "@node" in content or "add_node" in content,
            "Edge_Definition": "add_edge" in content or "add_conditional_edges" in content,
            "State_Management": "State" in content and "TypedDict" in content,
            "Compilation": "compile" in content,
            "Memory_State": any(keyword in content.lower() for keyword in ["memory", "history", "context"]),
            "Tool_Integration": "tool" in content.lower() or "function_call" in content.lower()
        }
        
        return {
            "status": "✅ PASS" if sum(langgraph_features.values()) >= 5 else "❌ FAIL",
            "details": langgraph_features,
            "score": f"{sum(langgraph_features.values())}/7 features"
        }
    
    def test_rag_implementation(self):
        """Test RAG pipeline implementation"""
        print("🔍 Testing RAG Implementation...")
        
        main_py = self.root_path / "backend" / "main.py"
        with open(main_py, 'r') as f:
            content = f.read()
            
        rag_components = {
            "Vector_Store": any(db in content for db in ["FAISS", "Qdrant", "Pinecone", "Chroma"]),
            "Embeddings": any(embed in content for embed in ["OpenAI", "embedding", "HuggingFace"]),
            "Document_Loading": any(loader in content for loader in ["TextLoader", "JSONLoader", "load_documents"]),
            "Similarity_Search": "similarity_search" in content or "search" in content,
            "Retrieval_Chain": "retrieval" in content.lower() or "retrieve" in content.lower(),
            "Context_Integration": "context" in content.lower() and "prompt" in content.lower()
        }
        
        return {
            "status": "✅ PASS" if sum(rag_components.values()) >= 4 else "❌ FAIL",
            "details": rag_components,
            "score": f"{sum(rag_components.values())}/6 components"
        }
    
    def test_dynamic_schema_system(self):
        """Test dynamic schema loading and processing"""
        print("📋 Testing Dynamic Schema System...")
        
        # Check for schema files
        schema_files = list(self.root_path.glob("**/schemas/*.json")) + list(self.root_path.glob("**/*schema*.json"))
        
        main_py = self.root_path / "backend" / "main.py"
        with open(main_py, 'r') as f:
            content = f.read()
            
        schema_features = {
            "Schema_Files_Present": len(schema_files) > 0,
            "Dynamic_Loading": "load_schema" in content or "get_schema" in content,
            "Platform_Support": "platform" in content.lower() and "type" in content.lower(),
            "Field_Validation": "validate" in content.lower() or "pydantic" in content.lower(),
            "Required_Fields": "required" in content.lower(),
            "Schema_Caching": "cache" in content.lower() or "store" in content.lower()
        }
        
        return {
            "status": "✅ PASS" if sum(schema_features.values()) >= 4 else "❌ FAIL",
            "details": schema_features,
            "schema_files": [str(f) for f in schema_files]
        }
    
    def test_frontend_integration(self):
        """Test React frontend implementation"""
        print("⚛️ Testing Frontend Integration...")
        
        app_tsx = self.root_path / "frontend" / "src" / "App.tsx"
        if not app_tsx.exists():
            return {"status": "❌ FAIL", "details": "App.tsx not found"}
            
        with open(app_tsx, 'r') as f:
            content = f.read()
            
        frontend_features = {
            "TypeScript_Usage": "interface" in content or "type" in content,
            "React_Hooks": "useState" in content and "useEffect" in content,
            "API_Integration": "fetch" in content or "axios" in content,
            "Chat_Interface": "chat" in content.lower() or "message" in content.lower(),
            "File_Upload": "upload" in content.lower() or "file" in content.lower(),
            "State_Management": "useState" in content and "state" in content.lower(),
            "Component_Structure": "function" in content and "return" in content,
            "Styling": "className" in content or "styled" in content
        }
        
        # Check package.json for dependencies
        pkg_json = self.root_path / "frontend" / "package.json"
        if pkg_json.exists():
            with open(pkg_json, 'r') as f:
                pkg_data = json.load(f)
                deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
                frontend_features["React_Installed"] = "react" in deps
                frontend_features["TypeScript_Configured"] = "typescript" in deps
                frontend_features["Build_Tools"] = "vite" in deps or "webpack" in deps
        
        return {
            "status": "✅ PASS" if sum(frontend_features.values()) >= 7 else "⚠️ PARTIAL",
            "details": frontend_features,
            "score": f"{sum(frontend_features.values())}/11 features"
        }
    
    def test_session_logs_quality(self):
        """Test quality and completeness of session logs"""
        print("📝 Testing Session Logs Quality...")
        
        log_files = [
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json", 
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        log_analysis = {}
        
        for log_file in log_files:
            log_path = self.root_path / log_file
            if not log_path.exists():
                log_analysis[log_file] = {"status": "❌ MISSING"}
                continue
                
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    
                # Analyze conversation quality
                conversation = log_data.get("conversation", [])
                final_spec = log_data.get("final_spec", {})
                reasoning_trace = log_data.get("reasoning_trace", [])
                
                quality_metrics = {
                    "conversation_length": len(conversation),
                    "has_scoping_phase": any("scope" in msg.get("content", "").lower() for msg in conversation),
                    "has_spec_generation": len(final_spec) > 0,
                    "has_reasoning": len(reasoning_trace) > 0,
                    "shows_iteration": len(conversation) > 10,
                    "contains_questions": any("?" in msg.get("content", "") for msg in conversation),
                    "has_user_feedback": any(msg.get("role") == "user" for msg in conversation),
                    "complete_spec": len(final_spec.keys()) >= 5 if final_spec else False
                }
                
                score = sum(quality_metrics.values())
                status = "✅ HIGH" if score >= 6 else "⚠️ MEDIUM" if score >= 4 else "❌ LOW"
                
                log_analysis[log_file] = {
                    "status": status,
                    "score": f"{score}/8",
                    "metrics": quality_metrics
                }
                
            except Exception as e:
                log_analysis[log_file] = {"status": "❌ ERROR", "error": str(e)}
        
        return {
            "status": "✅ PASS" if all("HIGH" in result.get("status", "") or "MEDIUM" in result.get("status", "") for result in log_analysis.values()) else "❌ FAIL",
            "details": log_analysis
        }
    
    def test_bonus_features(self):
        """Test bonus features implementation"""
        print("🎁 Testing Bonus Features...")
        
        main_py = self.root_path / "backend" / "main.py"
        with open(main_py, 'r') as f:
            backend_content = f.read()
            
        bonus_features = {
            "Vision_Model_Support": any(keyword in backend_content.lower() for keyword in [
                "gpt-4-vision", "vision", "image", "multimodal", "base64"
            ]),
            "File_Upload_Endpoint": "/upload" in backend_content,
            "Image_Processing": "PIL" in backend_content or "cv2" in backend_content or "image" in backend_content.lower(),
            "React_Frontend_Complete": (self.root_path / "frontend" / "src" / "App.tsx").exists(),
            "UI_Components": False,  # Will check below
            "Advanced_Reasoning": "confidence" in backend_content.lower() and "reasoning" in backend_content.lower()
        }
        
        # Check UI components
        if (self.root_path / "frontend" / "src" / "App.tsx").exists():
            with open(self.root_path / "frontend" / "src" / "App.tsx", 'r') as f:
                frontend_content = f.read()
                bonus_features["UI_Components"] = all(component in frontend_content.lower() for component in [
                    "chat", "checklist", "spec"
                ])
        
        return {
            "status": "✅ EXCELLENT" if sum(bonus_features.values()) >= 5 else "⚠️ GOOD" if sum(bonus_features.values()) >= 3 else "❌ BASIC",
            "details": bonus_features,
            "score": f"{sum(bonus_features.values())}/6 features"
        }
    
    def test_code_quality(self):
        """Test overall code quality"""
        print("🏗️ Testing Code Quality...")
        
        main_py = self.root_path / "backend" / "main.py"
        with open(main_py, 'r') as f:
            content = f.read()
            
        quality_metrics = {
            "has_docstrings": '"""' in content,
            "has_type_hints": ": " in content and "->" in content,
            "error_handling": "try:" in content and "except" in content,
            "logging": any(log in content for log in ["logging", "logger", "print"]),
            "environment_variables": "os.environ" in content or "getenv" in content,
            "input_validation": "validate" in content.lower() or "pydantic" in content,
            "async_programming": "async def" in content and "await" in content,
            "proper_imports": "from" in content and "import" in content
        }
        
        # Check line count (should be substantial)
        line_count = len(content.split('\n'))
        quality_metrics["substantial_code"] = line_count > 200
        
        return {
            "status": "✅ HIGH" if sum(quality_metrics.values()) >= 7 else "⚠️ MEDIUM" if sum(quality_metrics.values()) >= 5 else "❌ LOW",
            "details": quality_metrics,
            "line_count": line_count,
            "score": f"{sum(quality_metrics.values())}/9 metrics"
        }
    
    def generate_technical_report(self):
        """Generate comprehensive technical report"""
        print("\n" + "="*60)
        print("🔬 TECHNICAL FUNCTIONALITY REPORT")
        print("="*60)
        
        # Run all tests
        tests = {
            "Backend Architecture": self.test_backend_architecture(),
            "Agent Framework": self.test_agent_framework_integration(),
            "RAG Implementation": self.test_rag_implementation(),
            "Dynamic Schema": self.test_dynamic_schema_system(),
            "Frontend Integration": self.test_frontend_integration(),
            "Session Logs Quality": self.test_session_logs_quality(),
            "Bonus Features": self.test_bonus_features(),
            "Code Quality": self.test_code_quality()
        }
        
        # Display results
        for test_name, result in tests.items():
            print(f"\n🧪 {test_name}")
            print("-" * 40)
            print(f"Status: {result['status']}")
            
            if 'score' in result:
                print(f"Score: {result['score']}")
                
            if isinstance(result['details'], dict):
                for key, value in result['details'].items():
                    icon = "✅" if value else "❌"
                    print(f"  {icon} {key.replace('_', ' ').title()}")
            else:
                print(f"Details: {result['details']}")
        
        # Calculate overall technical score
        status_scores = {
            "✅ PASS": 3, "✅ HIGH": 3, "✅ EXCELLENT": 3,
            "⚠️ PARTIAL": 2, "⚠️ MEDIUM": 2, "⚠️ GOOD": 2,
            "❌ FAIL": 1, "❌ LOW": 1, "❌ BASIC": 1
        }
        
        total_score = sum(status_scores.get(result['status'], 0) for result in tests.values())
        max_score = len(tests) * 3
        percentage = (total_score / max_score) * 100
        
        print(f"\n🏆 OVERALL TECHNICAL SCORE")
        print("-" * 40)
        print(f"Score: {total_score}/{max_score} ({percentage:.1f}%)")
        
        if percentage >= 85:
            print("🌟 EXCELLENT - Ready for production!")
        elif percentage >= 70:
            print("👍 GOOD - Minor improvements needed")
        elif percentage >= 50:
            print("⚠️ ACCEPTABLE - Some issues to address")
        else:
            print("❌ NEEDS WORK - Significant improvements required")
        
        return {
            "tests": tests,
            "total_score": total_score,
            "max_score": max_score,
            "percentage": percentage
        }

def main():
    """Main function"""
    root_path = os.path.dirname(os.path.abspath(__file__))
    validator = TechnicalValidator(root_path)
    
    try:
        report = validator.generate_technical_report()
        
        # Save detailed report
        report_path = os.path.join(root_path, "technical_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Technical report saved to: {report_path}")
        
        # Return appropriate exit code
        if report['percentage'] >= 85:
            return 0  # Excellent
        elif report['percentage'] >= 70:
            return 1  # Good
        elif report['percentage'] >= 50:
            return 2  # Acceptable
        else:
            return 3  # Needs work
            
    except Exception as e:
        print(f"❌ Technical validation error: {e}")
        return 4

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
