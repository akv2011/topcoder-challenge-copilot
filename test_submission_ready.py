#!/usr/bin/env python3
"""
AI Copilot Agent - Pre-Submission Testing Suite
This script performs comprehensive testing to ensure the system is submission-ready.
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path

class PreSubmissionTester:
    def __init__(self):
        self.root_path = Path.cwd()
        self.results = {
            "tests_passed": [],
            "tests_failed": [],
            "warnings": [],
            "submission_ready": False
        }
    
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("📦 Checking Dependencies...")
        print("-" * 40)
        
        # Check Python packages
        required_packages = [
            "fastapi", "uvicorn", "openai", "langchain", 
            "langgraph", "pydantic", "faiss-cpu", "sqlite3"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} - NOT INSTALLED")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️ Install missing packages: pip install {' '.join(missing_packages)}")
            self.results["warnings"].append(f"Missing packages: {missing_packages}")
            return False
        
        self.results["tests_passed"].append("Dependencies check")
        return True
    
    def check_environment_setup(self):
        """Check environment variables and configuration"""
        print("\n🔧 Checking Environment Setup...")
        print("-" * 40)
        
        # Check for OpenAI API key
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            print(f"  ✅ OPENAI_API_KEY found (ends with: ...{openai_key[-4:]})")
        else:
            print("  ❌ OPENAI_API_KEY not found")
            print("     Set it with: export OPENAI_API_KEY='your-key-here'")
            self.results["tests_failed"].append("OpenAI API key not set")
            return False
        
        # Check backend files
        backend_main = self.root_path / "backend" / "main.py"
        if backend_main.exists():
            print("  ✅ Backend main.py exists")
        else:
            print("  ❌ Backend main.py missing")
            self.results["tests_failed"].append("Backend main.py missing")
            return False
        
        self.results["tests_passed"].append("Environment setup")
        return True
    
    def test_backend_startup(self):
        """Test backend server startup"""
        print("\n🚀 Testing Backend Startup...")
        print("-" * 40)
        
        try:
            # Start backend process
            process = subprocess.Popen(
                ["python", "main.py"],
                cwd=self.root_path / "backend",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for startup
            time.sleep(5)
            
            # Test health endpoint
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print("  ✅ Backend started successfully")
                    print("  ✅ Health endpoint responding")
                    
                    # Test basic endpoints
                    endpoints = ["/", "/sessions", "/schemas/topcoder/design"]
                    for endpoint in endpoints:
                        try:
                            resp = requests.get(f"http://localhost:8000{endpoint}", timeout=3)
                            status = "✅" if resp.status_code in [200, 404] else "❌"
                            print(f"  {status} Endpoint {endpoint} ({resp.status_code})")
                        except:
                            print(f"  ⚠️ Endpoint {endpoint} timeout")
                    
                    process.terminate()
                    process.wait()
                    
                    self.results["tests_passed"].append("Backend startup")
                    return True
                else:
                    print(f"  ❌ Health endpoint failed: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                print("  ❌ Backend not responding")
            
            process.terminate()
            process.wait()
            
        except Exception as e:
            print(f"  ❌ Backend startup error: {e}")
        
        self.results["tests_failed"].append("Backend startup")
        return False
    
    def test_core_functionality(self):
        """Test core AI agent functionality"""
        print("\n🤖 Testing Core AI Functionality...")
        print("-" * 40)
        
        # Start backend for testing
        process = subprocess.Popen(
            ["python", "main.py"],
            cwd=self.root_path / "backend",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        time.sleep(5)
        
        try:
            # Test session creation
            session_response = requests.post(
                "http://localhost:8000/sessions",
                json={"user_goal": "Test challenge creation"},
                timeout=10
            )
            
            if session_response.status_code == 201:
                print("  ✅ Session creation works")
                
                session_id = session_response.json()["session_id"]
                
                # Test chat functionality
                chat_response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "session_id": session_id,
                        "message": "I want to create a mobile app design challenge"
                    },
                    timeout=30
                )
                
                if chat_response.status_code == 200:
                    print("  ✅ Chat functionality works")
                    ai_response = chat_response.json().get("response", "")
                    
                    if len(ai_response) > 50:
                        print(f"  ✅ AI response generated ({len(ai_response)} chars)")
                        
                        # Test RAG search
                        try:
                            rag_response = requests.post(
                                "http://localhost:8000/search",
                                json={"query": "mobile app design"},
                                timeout=15
                            )
                            
                            if rag_response.status_code == 200:
                                print("  ✅ RAG search functionality works")
                                self.results["tests_passed"].append("Core functionality")
                                return True
                            else:
                                print("  ⚠️ RAG search not working")
                                
                        except:
                            print("  ⚠️ RAG search timeout")
                    else:
                        print("  ❌ AI response too short")
                else:
                    print(f"  ❌ Chat failed: {chat_response.status_code}")
            else:
                print(f"  ❌ Session creation failed: {session_response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Core functionality error: {e}")
        finally:
            process.terminate()
            process.wait()
        
        self.results["tests_failed"].append("Core functionality")
        return False
    
    def test_session_logs_quality(self):
        """Test quality of existing session logs"""
        print("\n📝 Testing Session Logs Quality...")
        print("-" * 40)
        
        log_files = [
            "session_log_1_design_challenge__mobile_app_ui.json",
            "session_log_2_development_challenge__api_backend.json",
            "session_log_3_innovation_challenge__sustainability.json"
        ]
        
        valid_logs = 0
        
        for log_file in log_files:
            log_path = self.root_path / log_file
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        log_data = json.load(f)
                    
                    conversation = log_data.get("conversation", [])
                    final_spec = log_data.get("final_spec", {})
                    reasoning_trace = log_data.get("reasoning_trace", [])
                    
                    if len(conversation) >= 5 and len(final_spec) > 0 and len(reasoning_trace) > 0:
                        print(f"  ✅ {log_file} - Valid structure")
                        valid_logs += 1
                    else:
                        print(f"  ⚠️ {log_file} - Incomplete structure")
                        
                except Exception as e:
                    print(f"  ❌ {log_file} - Error: {e}")
            else:
                print(f"  ❌ {log_file} - Missing")
        
        if valid_logs >= 3:
            print(f"  ✅ All {valid_logs} session logs are valid")
            self.results["tests_passed"].append("Session logs quality")
            return True
        else:
            print(f"  ❌ Only {valid_logs}/3 session logs are valid")
            self.results["tests_failed"].append("Session logs quality")
            return False
    
    def test_frontend_build(self):
        """Test frontend build process"""
        print("\n⚛️ Testing Frontend Build...")
        print("-" * 40)
        
        frontend_dir = self.root_path / "frontend"
        if not frontend_dir.exists():
            print("  ❌ Frontend directory not found")
            self.results["tests_failed"].append("Frontend build")
            return False
        
        try:
            # Check if node_modules exists, if not install
            if not (frontend_dir / "node_modules").exists():
                print("  📦 Installing npm dependencies...")
                result = subprocess.run(
                    ["npm", "install"],
                    cwd=frontend_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    print(f"  ❌ npm install failed: {result.stderr}")
                    self.results["tests_failed"].append("Frontend dependencies")
                    return False
                else:
                    print("  ✅ npm dependencies installed")
            
            # Test build
            print("  🔨 Testing build process...")
            build_result = subprocess.run(
                ["npm", "run", "build"],
                cwd=frontend_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if build_result.returncode == 0:
                print("  ✅ Frontend builds successfully")
                self.results["tests_passed"].append("Frontend build")
                return True
            else:
                print(f"  ❌ Frontend build failed: {build_result.stderr}")
                self.results["tests_failed"].append("Frontend build")
                return False
                
        except subprocess.TimeoutExpired:
            print("  ❌ Frontend build timeout")
            self.results["tests_failed"].append("Frontend build timeout")
            return False
        except Exception as e:
            print(f"  ❌ Frontend build error: {e}")
            self.results["tests_failed"].append("Frontend build error")
            return False
    
    def generate_submission_checklist(self):
        """Generate final submission checklist"""
        print(f"\n{'='*60}")
        print("📋 SUBMISSION READINESS CHECKLIST")
        print(f"{'='*60}")
        
        # Required files check
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
        
        print("\n📁 Required Files:")
        all_files_present = True
        for file_path in required_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print(f"  ✅ {file_path} ({size:,} bytes)")
            else:
                print(f"  ❌ {file_path} - MISSING")
                all_files_present = False
        
        # Test results summary
        print(f"\n🧪 Test Results:")
        for test in self.results["tests_passed"]:
            print(f"  ✅ {test}")
        
        for test in self.results["tests_failed"]:
            print(f"  ❌ {test}")
        
        if self.results["warnings"]:
            print(f"\n⚠️ Warnings:")
            for warning in self.results["warnings"]:
                print(f"  • {warning}")
        
        # Overall assessment
        total_tests = len(self.results["tests_passed"]) + len(self.results["tests_failed"])
        passed_tests = len(self.results["tests_passed"])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 Overall Score: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Determine submission readiness
        self.results["submission_ready"] = (
            all_files_present and 
            success_rate >= 80 and 
            len(self.results["tests_failed"]) <= 1
        )
        
        if self.results["submission_ready"]:
            print("\n🎯 ASSESSMENT: ✅ READY FOR SUBMISSION!")
            print("\n🚀 Next Steps:")
            print("  1. Start backend: cd backend && python main.py")
            print("  2. Start frontend: cd frontend && npm run dev")
            print("  3. Test the full system manually")
            print("  4. Submit to Topcoder!")
        else:
            print("\n🎯 ASSESSMENT: ❌ NOT READY - FIX ISSUES FIRST")
            print("\n🔧 Required Actions:")
            for test in self.results["tests_failed"]:
                print(f"  • Fix: {test}")
        
        # Save results
        with open("submission_test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: submission_test_results.json")
        
        return self.results["submission_ready"]
    
    def run_all_tests(self):
        """Run complete pre-submission test suite"""
        print("🎯 AI COPILOT AGENT - PRE-SUBMISSION TESTING")
        print("="*60)
        print(f"📅 Testing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Run all tests
        self.check_dependencies()
        self.check_environment_setup()
        self.test_backend_startup()
        self.test_core_functionality()
        self.test_session_logs_quality()
        self.test_frontend_build()
        
        # Generate final assessment
        return self.generate_submission_checklist()

def main():
    """Main testing function"""
    tester = PreSubmissionTester()
    
    try:
        ready = tester.run_all_tests()
        return 0 if ready else 1
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        return 2
    except Exception as e:
        print(f"❌ Testing error: {e}")
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
