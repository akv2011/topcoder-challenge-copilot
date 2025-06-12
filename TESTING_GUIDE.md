# 🧪 MANUAL TESTING GUIDE - AI Copilot Agent

## 🎯 How to Test Your Implementation Before Submission

This guide walks you through **actual testing** of the system to verify everything works correctly.

---

## ⚡ QUICK AUTOMATED TESTING

### 1. Run Pre-Submission Tests
```bash
python3 test_submission_ready.py
```
**This will:**
- ✅ Check all dependencies
- ✅ Verify environment setup
- ✅ Test backend startup
- ✅ Test core AI functionality
- ✅ Validate session logs
- ✅ Test frontend build
- ✅ Generate submission readiness report

### 2. Run Live Functionality Tests
```bash
python3 test_live_functionality.py
```
**This will:**
- 🚀 Start backend automatically
- 🧪 Test all API endpoints
- 💬 Test chat functionality
- 🔍 Test RAG search
- 📋 Test schema loading
- 📁 Test file upload
- 🔄 Test complete user workflow

---

## 🚀 MANUAL TESTING STEPS

### Step 1: Start the Backend
```bash
cd backend
python main.py
```
**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**✅ Test:** Open http://localhost:8000 in browser - should see welcome message

### Step 2: Test API Endpoints

#### A. Health Check
```bash
curl http://localhost:8000/health
```
**Expected:** `{"status": "healthy"}`

#### B. Create Session
```bash
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"user_goal": "I want to create a mobile app design challenge"}'
```
**Expected:** `{"session_id": "some-uuid", "message": "Session created"}`

#### C. Test Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "your-session-id", "message": "I want to focus on the login screen design"}'
```
**Expected:** Long AI response with questions about the design challenge

#### D. Test Schema Loading
```bash
curl http://localhost:8000/schemas/topcoder/design
```
**Expected:** JSON schema with fields like title, overview, requirements, etc.

#### E. Test RAG Search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "mobile app design UI UX"}'
```
**Expected:** Array of relevant challenge examples

### Step 3: Start the Frontend
```bash
cd frontend
npm install
npm run dev
```
**Expected Output:**
```
  VITE v4.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

**✅ Test:** Open http://localhost:5173 - should see React app with chat interface

### Step 4: Test Complete User Flow

#### A. Open Frontend (http://localhost:5173)
1. **See chat interface** ✅
2. **Type message:** "I want to create a food delivery app"
3. **AI should respond** with scoping questions ✅
4. **Continue conversation** for 5-10 exchanges ✅
5. **See final challenge spec** generated ✅

#### B. Test File Upload (Bonus Feature)
1. **Click upload button** in frontend ✅
2. **Upload an image file** ✅
3. **AI should analyze** and incorporate into conversation ✅

### Step 5: Verify Session Logs

**Check that these files exist and have valid content:**
```bash
ls -la session_log_*.json
```

**Validate structure:**
```bash
python3 -c "
import json
for i in [1,2,3]:
    with open(f'session_log_{i}_*.json', 'r') as f:
        data = json.load(f)
        print(f'Log {i}: {len(data[\"conversation\"])} messages, {len(data[\"final_spec\"])} spec fields')
"
```

---

## 🔍 TESTING SCENARIOS

### Scenario 1: Design Challenge
1. **Input:** "I want to design a mobile banking app"
2. **Expected Flow:**
   - AI asks about project stage
   - AI suggests focusing on specific screens
   - AI asks about design style, timeline
   - AI generates design challenge spec

### Scenario 2: Development Challenge  
1. **Input:** "I need to build an API for my e-commerce site"
2. **Expected Flow:**
   - AI asks about tech stack preferences
   - AI clarifies API requirements
   - AI suggests development milestones
   - AI generates development challenge spec

### Scenario 3: Innovation Challenge
1. **Input:** "I want to solve climate change with technology"
2. **Expected Flow:**
   - AI helps narrow down scope
   - AI suggests specific problem areas
   - AI asks about solution approach
   - AI generates innovation challenge spec

---

## ✅ SUCCESS CRITERIA

Your system is working correctly if:

### Backend ✅
- ✅ Server starts without errors
- ✅ All endpoints respond correctly
- ✅ AI generates contextual responses
- ✅ RAG search returns relevant results
- ✅ Sessions persist properly
- ✅ File upload works (bonus)

### Frontend ✅
- ✅ React app loads and renders
- ✅ Chat interface is functional
- ✅ Messages send and receive properly
- ✅ File upload UI works (bonus)
- ✅ Challenge spec displays correctly

### AI Agent ✅
- ✅ Asks relevant scoping questions
- ✅ Doesn't impose scope on user
- ✅ Adapts based on user responses
- ✅ Generates complete challenge specs
- ✅ Provides reasoning for decisions
- ✅ Learns from user feedback

---

## 🚨 TROUBLESHOOTING

### Backend Won't Start
```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python main.py
```

### Frontend Won't Start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### AI Not Responding
- Check OpenAI API key is set
- Check internet connection
- Check console logs for errors

### Tests Failing
```bash
# Run comprehensive diagnostics
python3 test_submission_ready.py

# Check specific issues
python3 test_live_functionality.py
```

---

## 🎯 SUBMISSION CHECKLIST

Before submitting, ensure:

### ✅ Files Present
- [ ] `backend/main.py` exists and works
- [ ] `frontend/src/App.tsx` exists and builds
- [ ] 3 session log JSON files exist
- [ ] `README.md` is complete

### ✅ Functionality Working  
- [ ] Backend starts successfully
- [ ] Chat generates AI responses
- [ ] RAG search returns results
- [ ] Frontend renders correctly
- [ ] Complete user workflow works

### ✅ Requirements Met
- [ ] Uses LangGraph for agent orchestration
- [ ] Uses FAISS for vector storage
- [ ] Uses FastAPI for backend
- [ ] Implements dynamic schema loading
- [ ] Includes reasoning traces
- [ ] Supports feedback loop

**🚀 If all checks pass, you're ready to submit!**
