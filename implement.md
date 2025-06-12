Challenge Overview
This is Phase 1 of a 3-part program to build an AI Copilot Agent that helps users define and launch structured innovation or development challenges on platforms like Topcoder, HeroX, Kaggle, Zindi, or internal company systems.

Your mission is to implement the reasoning and guidance engine of the Copilot — the backend AI logic that interacts with a user to understand their goal, assess its feasibility for a challenge, clarify details, and ultimately construct a well-scoped, structured spec ready for submission.

The AI agent must ask dynamic, adaptive questions, guide scoping with reasoning, and refine its behavior based on user answers and platform requirements. No hardcoded scripts — the system should reason, recall, and respond contextually.

Objectives
Your Copilot must be able to:

Accept a high-level user input (e.g. “I want to build an app for cleaning services”)
Initiate a scoping dialogue:
Ask contextual questions about project status
Clarify whether this is design, development, testing, or PoC
Suggest possible scoping directions
Let the user confirm or reject them
Reassess scope feasibility iteratively
Once scope is confirmed, load a dynamic schema (platform-specific challenge fields)
Drive structured Q&A to complete the spec fields
Retrieve similar challenges via RAG to inform questions and structure
Output a structured, reasoned spec (JSON or similar)
Store and adapt to user feedback per section or suggestion
How Scoping Should Work
The Copilot is not allowed to impose a scope.

Instead, it should:

Evaluate scope feasibility
Example prompt:
"This seems like a broad goal. Can you tell me what part is currently your priority, or what’s already done?"

Ask project clarifying questions
“What stage is the project in?”
“Do you have mockups or designs?”
“Is this the first task, or part of a series of challenges?”
Propose possible scoped ideas based on the user's reply, e.g.:
“Based on what you shared, we might start with designing the booking screen. Does that work for you?”

If user says yes → continue
If user says no → ask what they prefer, then re-evaluate if that fits challenge boundaries (timeline, cost, deliverable size)
Repeat the cycle until both agent and user agree on a scoped unit of work
Then start compiling the spec with dynamic field-driven questions
Use of Agent Frameworks
You must use an existing AI agent orchestration framework:

Recommended:

LangGraph
CrewAI
AutoGen
SuperAgent
Flowise
You may also use OpenAI Assistants API if it includes:

Memory across turns
Function calling
Tool-based interaction
Dynamic Field Schema (Configurable)
Challenge requirements vary per platform. Your Copilot must:

Load challenge field definitions dynamically (JSON, YAML, or API)
Use them to:
Ask relevant questions
Track checklist progress
Format the final output
Example (suggested, not required):

{
  "platform": "Topcoder",
  "challenge_type": "Design",
  "fields": {
    "title": { "required": true },
    "overview": { "required": true },
    "objectives": { "required": true },
    "tech_stack": { "required": false },
    "timeline": { "required": true },
    "prize_structure": { "required": true },
    "judging_criteria": { "required": true }
  }
}
Fields must not be hardcoded. Make the schema configurable per platform and challenge type.

RAG Integration
Integrate a Retrieval-Augmented Generation (RAG) pipeline:

Use a vector database (e.g., FAISS, Pinecone, Qdrant)
Store past challenge specs, templates, and suggestions
Retrieve relevant examples based on user topic, category, or phase
Use these as grounding for:
Question phrasing
Estimating timeline
Suggesting judging criteria
Platform-specific formatting
Feedback Loop Handling
As users answer questions or reject suggestions:

Capture their feedback (e.g., “React won’t work for us”)
Adapt future prompts during the session to avoid repetition
Log reasoning traces for all suggestions
If rejected → ask why → re-suggest
Example:

{
  "field": "tech_stack",
  "ai_suggestion": "React",
  "accepted": false,
  "reason": "We prefer Flutter for cross-platform",
  "prompt_update": true
}
Output Format (Structured Spec + Reasoning)
Final spec output should include:

Structured challenge fields (e.g., JSON or object)
Reasoning per field (why it was filled this way)
Confidence level (optional)
Example:

{
  "title": "Cleaner Booking Flow - UI Prototype",
  "overview": "Design 3 screens for customer booking in a cleaning app",
  "tech_stack": ["Figma"],
  "timeline": { "submission_days": 6 },
  "reasoning_trace": [
    {
      "field": "tech_stack",
      "source": "User confirmed mockup design only; no code needed",
      "confidence": 0.92
    }
  ]
}
Tech Stack
LLM: OpenAI GPT-4 (with Functions or Assistants API)
Agent Framework: LangGraph (preferred) or CrewAI
Prompt Routing & Memory: LangChain (optional if using LangGraph)
RAG Framework: LangChain or LlamaIndex
Vector Store: Qdrant (preferred), Pinecone, or FAISS (local dev)
Embeddings: OpenAI text-embedding-3-small or Instructor-XL
Server: FastAPI (Python) or Node.js + Express
Deliverables
Agent backend with:
Scoping dialogue logic
Schema-aware Q&A logic
RAG integration
Reasoning trace generation
3 end-to-end session logs (user → Copilot → spec)
README explaining:
Agent orchestration setup
Memory handling
Prompt strategies
Schema configurability
Bonus Points
You’ll earn bonus points in functionality and UI evaluation criterias for:

Vision model support (upload mockups for context)
⚛React frontend integration (we provide the UI shell):
Chat interface
Checklist view
Challenge spec viewer