#!/usr/bin/env python3
"""
AI Copilot Agent - Live Functionality Testing
This script actually tests the running system to verify it works correctly.
"""

import requests
import json
import time
import os
import subprocess
import signal
from pathlib import Path

class LiveFunctionalityTester:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.backend_process = None
        self.test_results = {}
        
    def start_backend(self):
        """Start the backend server"""
        print("🚀 Starting backend server...")
        
        backend_dir = Path("backend")
        if not backend_dir.exists():
            print("❌ Backend directory not found")
            return False
            
        try:
            # Start backend in background
            self.backend_process = subprocess.Popen(
                ["python", "main.py"],
                cwd=backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            for i in range(10):
                try:
                    response = requests.get(f"{self.backend_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Backend server started successfully")
                        return True
                except:
                    time.sleep(1)
                    
            print("❌ Backend server failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Error starting backend: {e}")
            return False
    
    def stop_backend(self):
        """Stop the backend server"""
        if self.backend_process:
            print("🛑 Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except:
                self.backend_process.kill()
            
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print("\n🔍 Testing API Endpoints...")
        print("-" * 40)
        
        endpoints_to_test = [
            ("GET", "/health", None, "Health check"),
            ("GET", "/", None, "Root endpoint"),
            ("GET", "/sessions", None, "List sessions"),
            ("GET", "/schemas/topcoder/design", None, "Get schema"),
            ("POST", "/sessions", {"user_goal": "Test challenge"}, "Create session"),
        ]
        
        for method, endpoint, data, description in endpoints_to_test:
            try:
                url = f"{self.backend_url}{endpoint}"
                
                if method == "GET":
                    response = requests.get(url, timeout=10)
                elif method == "POST":
                    response = requests.post(url, json=data, timeout=10)
                
                status = "✅ PASS" if response.status_code in [200, 201] else "❌ FAIL"
                print(f"  {status} {method} {endpoint} - {description} ({response.status_code})")
                
                self.test_results[f"{method}_{endpoint}"] = {
                    "status": response.status_code,
                    "success": response.status_code in [200, 201]
                }
                
            except Exception as e:
                print(f"  ❌ FAIL {method} {endpoint} - {description} (Error: {e})")
                self.test_results[f"{method}_{endpoint}"] = {
                    "status": "error",
                    "success": False,
                    "error": str(e)
                }
    
    def test_chat_functionality(self):
        """Test the chat/conversation functionality"""
        print("\n💬 Testing Chat Functionality...")
        print("-" * 40)
        
        try:
            # Create a new session
            session_response = requests.post(
                f"{self.backend_url}/sessions",
                json={"user_goal": "I want to create a mobile app for food delivery"},
                timeout=30
            )
            
            if session_response.status_code != 201:
                print("❌ Failed to create session")
                return False
                
            session_data = session_response.json()
            session_id = session_data["session_id"]
            print(f"✅ Created session: {session_id}")
            
            # Test chat interaction
            chat_response = requests.post(
                f"{self.backend_url}/chat",
                json={
                    "session_id": session_id,
                    "message": "Yes, I want to focus on the user interface design"
                },
                timeout=30
            )
            
            if chat_response.status_code == 200:
                chat_data = chat_response.json()
                print(f"✅ Chat response received: {len(chat_data.get('response', ''))} characters")
                print(f"   Preview: {chat_data.get('response', '')[:100]}...")
                
                self.test_results["chat_functionality"] = {
                    "success": True,
                    "session_id": session_id,
                    "response_length": len(chat_data.get('response', ''))
                }
                return True
            else:
                print(f"❌ Chat request failed: {chat_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Chat functionality error: {e}")
            self.test_results["chat_functionality"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_rag_functionality(self):
        """Test RAG (vector search) functionality"""
        print("\n🔍 Testing RAG Functionality...")
        print("-" * 40)
        
        try:
            # Test RAG search
            rag_response = requests.post(
                f"{self.backend_url}/search",
                json={"query": "mobile app design challenge"},
                timeout=20
            )
            
            if rag_response.status_code == 200:
                rag_data = rag_response.json()
                results = rag_data.get("results", [])
                print(f"✅ RAG search returned {len(results)} results")
                
                if results:
                    print(f"   Top result: {results[0].get('title', 'No title')[:50]}...")
                
                self.test_results["rag_functionality"] = {
                    "success": True,
                    "results_count": len(results)
                }
                return True
            else:
                print(f"❌ RAG search failed: {rag_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ RAG functionality error: {e}")
            self.test_results["rag_functionality"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_schema_loading(self):
        """Test dynamic schema loading"""
        print("\n📋 Testing Schema Loading...")
        print("-" * 40)
        
        platforms = ["topcoder", "kaggle", "herox"]
        challenge_types = ["design", "development", "data_science"]
        
        for platform in platforms:
            for challenge_type in challenge_types:
                try:
                    response = requests.get(
                        f"{self.backend_url}/schemas/{platform}/{challenge_type}",
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        schema_data = response.json()
                        fields_count = len(schema_data.get("fields", {}))
                        print(f"✅ {platform}/{challenge_type}: {fields_count} fields")
                    else:
                        print(f"⚠️ {platform}/{challenge_type}: Not available ({response.status_code})")
                        
                except Exception as e:
                    print(f"❌ {platform}/{challenge_type}: Error - {e}")
    
    def test_file_upload(self):
        """Test file upload functionality (vision model)"""
        print("\n📁 Testing File Upload...")
        print("-" * 40)
        
        try:
            # Create a dummy image file for testing
            test_content = b"Test image content"
            
            files = {"file": ("test_image.jpg", test_content, "image/jpeg")}
            data = {"session_id": "test_session"}
            
            response = requests.post(
                f"{self.backend_url}/upload",
                files=files,
                data=data,
                timeout=20
            )
            
            if response.status_code == 200:
                upload_data = response.json()
                print(f"✅ File upload successful")
                print(f"   Response: {upload_data.get('message', 'No message')}")
                
                self.test_results["file_upload"] = {
                    "success": True,
                    "response": upload_data
                }
                return True
            else:
                print(f"❌ File upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ File upload error: {e}")
            self.test_results["file_upload"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_full_workflow(self):
        """Test complete user workflow"""
        print("\n🔄 Testing Complete Workflow...")
        print("-" * 40)
        
        try:
            # 1. Create session
            print("Step 1: Creating session...")
            session_response = requests.post(
                f"{self.backend_url}/sessions",
                json={"user_goal": "I want to build a weather app"},
                timeout=30
            )
            
            if session_response.status_code != 201:
                print("❌ Session creation failed")
                return False
                
            session_id = session_response.json()["session_id"]
            print(f"✅ Session created: {session_id}")
            
            # 2. Scoping dialogue
            print("Step 2: Scoping dialogue...")
            chat_steps = [
                "I want to focus on mobile app design",
                "Yes, let's start with the main weather display screen",
                "I prefer a modern, minimalist design",
                "The timeline should be about 1 week"
            ]
            
            for i, message in enumerate(chat_steps):
                print(f"  User message {i+1}: {message[:30]}...")
                
                response = requests.post(
                    f"{self.backend_url}/chat",
                    json={
                        "session_id": session_id,
                        "message": message
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    ai_response = response.json().get("response", "")
                    print(f"  AI response {i+1}: {ai_response[:50]}...")
                else:
                    print(f"  ❌ Chat step {i+1} failed")
                    return False
            
            # 3. Get final spec
            print("Step 3: Getting final specification...")
            spec_response = requests.get(
                f"{self.backend_url}/sessions/{session_id}/spec",
                timeout=20
            )
            
            if spec_response.status_code == 200:
                spec_data = spec_response.json()
                print(f"✅ Final spec generated with {len(spec_data)} fields")
                
                self.test_results["full_workflow"] = {
                    "success": True,
                    "session_id": session_id,
                    "spec_fields": len(spec_data)
                }
                return True
            else:
                print(f"❌ Spec generation failed: {spec_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Workflow test error: {e}")
            self.test_results["full_workflow"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def check_frontend_accessibility(self):
        """Check if frontend is accessible"""
        print("\n⚛️ Checking Frontend Accessibility...")
        print("-" * 40)
        
        try:
            response = requests.get(self.frontend_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ Frontend accessible at {self.frontend_url}")
                return True
            else:
                print(f"⚠️ Frontend returned status {response.status_code}")
                return False
        except:
            print(f"⚠️ Frontend not running at {self.frontend_url}")
            print("   To start frontend: cd frontend && npm install && npm run dev")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print(f"\n{'='*60}")
        print("🏆 LIVE FUNCTIONALITY TEST REPORT")
        print(f"{'='*60}")
        
        # Calculate overall success rate
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get("success", False))
        
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📊 Overall Success Rate: {successful_tests}/{total_tests} ({success_rate:.1f}%)")
        
        # Detailed results
        print(f"\n📋 Detailed Test Results:")
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅" if result.get("success", False) else "❌"
                print(f"  {status} {test_name.replace('_', ' ').title()}")
                if "error" in result:
                    print(f"     Error: {result['error']}")
            else:
                print(f"  ⚠️ {test_name}: {result}")
        
        # Recommendations
        print(f"\n🎯 Recommendations:")
        if success_rate >= 90:
            print("✅ Excellent! System is fully functional and ready for submission.")
        elif success_rate >= 70:
            print("⚠️ Good functionality with minor issues. Review failed tests.")
        else:
            print("❌ Significant issues found. Fix critical errors before submission.")
        
        # Save report
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "test_results": self.test_results
        }
        
        with open("live_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Report saved to: live_test_report.json")
        return report_data
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🎯 AI COPILOT AGENT - LIVE FUNCTIONALITY TESTING")
        print("="*60)
        
        try:
            # Start backend
            if not self.start_backend():
                print("❌ Cannot start backend - aborting tests")
                return False
            
            # Run tests
            self.test_api_endpoints()
            self.test_chat_functionality()
            self.test_rag_functionality()
            self.test_schema_loading()
            self.test_file_upload()
            self.test_full_workflow()
            self.check_frontend_accessibility()
            
            # Generate report
            report = self.generate_test_report()
            
            return report["success_rate"] >= 70
            
        finally:
            # Always stop backend
            self.stop_backend()

def main():
    """Main test function"""
    tester = LiveFunctionalityTester()
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        tester.stop_backend()
        return 2
    except Exception as e:
        print(f"❌ Testing error: {e}")
        tester.stop_backend()
        return 3

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
