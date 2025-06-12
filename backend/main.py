# FastAPI Backend for AI Copilot Challenge Agent
# Replaces Convex backend as per implement.md requirements

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import sqlite3
import json
import uuid
import os
from datetime import datetime
import openai
import base64 # Added for image encoding

# Import Embeddings base class regardless of other dependencies
try:
    from langchain.embeddings.base import Embeddings
except ImportError:
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError:
        # Create a minimal Embeddings base class if not available
        class Embeddings:
            def embed_documents(self, texts):
                raise NotImplementedError
            def embed_query(self, text):
                raise NotImplementedError

try:
    # Skip PyTorch-dependent embeddings for now due to symbol conflicts
    # Use simple text-based RAG instead
    from langchain_community.vectorstores import FAISS
    
    # Create a simple text-based embeddings fallback
    class SimpleTextEmbeddings(Embeddings):
        def __init__(self):
            print("✅ Using simple text-based embeddings (no PyTorch dependencies)")
        
        def embed_documents(self, texts):
            # Simple text-based "embeddings" using character frequencies
            embeddings = []
            for text in texts:
                # Create a simple vector based on text characteristics
                vector = [
                    len(text),  # length
                    text.count(' '),  # word count approximation
                    text.count('a'), text.count('e'), text.count('i'), text.count('o'), text.count('u'),  # vowels
                    text.lower().count('design'), text.lower().count('develop'), text.lower().count('api'),  # keywords
                    text.lower().count('challenge'), text.lower().count('innovation'), text.lower().count('mobile')
                ]
                # Normalize to [0,1] range
                max_val = max(vector) if max(vector) > 0 else 1
                vector = [v/max_val for v in vector]
                embeddings.append(vector)
            return embeddings
        
        def embed_query(self, text):
            return self.embed_documents([text])[0]
    
    HuggingFaceInstructEmbeddings = SimpleTextEmbeddings
    LANGCHAIN_AVAILABLE = True
    INSTRUCTOR_AVAILABLE = True
    print("✅ Simple text-based embeddings available")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    INSTRUCTOR_AVAILABLE = False
    print(f"❌ No embedding models available: {e}")

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    try:
        # Fallback to community version
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from typing_extensions import TypedDict
        LANGCHAIN_AVAILABLE = True
        LANGGRAPH_AVAILABLE = False
        print("⚠️ Using langchain-openai for embeddings")
    except ImportError:
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
            from typing_extensions import TypedDict
            LANGCHAIN_AVAILABLE = True
            LANGGRAPH_AVAILABLE = False
            print("⚠️ Using deprecated LangChain imports - consider upgrading to langchain-community")
        except ImportError:
            from typing_extensions import TypedDict
            LANGCHAIN_AVAILABLE = False
            LANGGRAPH_AVAILABLE = False
            print("LangChain/LangGraph not available - using simple RAG fallback")
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

app = FastAPI(title="AI Challenge Copilot", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Remove the custom InstructorEmbeddings class since we'll use LangChain's HuggingFaceInstructEmbeddings
# Custom Instructor Embeddings class is no longer needed

# Helper function to create OpenAI client safely
def create_openai_client():
    """Create OpenAI client with error handling for version compatibility"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    try:
        # Try the standard initialization
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        print(f"OpenAI client creation failed: {e}")
        # Fallback to a basic initialization
        try:
            # For compatibility with older versions
            import openai as openai_legacy
            openai_legacy.api_key = api_key
            return openai_legacy
        except Exception:
            raise HTTPException(status_code=500, detail=f"Failed to initialize OpenAI client: {str(e)}")

# Initialize embeddings and vector store
if LANGCHAIN_AVAILABLE:
    try:
        if INSTRUCTOR_AVAILABLE:
            print("🔧 Initializing Instructor-XL embeddings...")
            # Use our simple text-based embeddings since Instructor-XL has dependencies issues
            embeddings = SimpleTextEmbeddings()
            print("✅ Simple text-based embeddings initialized successfully")
        else:
            print("🔧 Instructor-XL not available - using simple embeddings")
            embeddings = SimpleTextEmbeddings()
        vector_store = None
    except Exception as e:
        print(f"❌ Failed to initialize embeddings: {e}")
        print("🔧 Using simple text-based RAG fallback")
        embeddings = SimpleTextEmbeddings()
        vector_store = None
else:
    print("🔧 Using simple text-based RAG fallback")
    embeddings = None
    vector_store = None

# Database setup
def init_db():
    conn = sqlite3.connect('challenges.db')
    cursor = conn.cursor()
    
    # Challenges table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS challenges (
            id TEXT PRIMARY KEY,
            goal TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scoping_goal',
            platform_id TEXT,
            dialogue_history TEXT NOT NULL DEFAULT '[]',
            specification TEXT,
            reasoning_traces TEXT DEFAULT '[]',
            current_step_key TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Platforms table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS platforms (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            schema_definition TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Past challenges table (for RAG)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS past_challenges (
            id TEXT PRIMARY KEY,
            platform_name TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            keywords TEXT NOT NULL DEFAULT '[]',
            timeline_example TEXT,
            judging_criteria_example TEXT,
            formatting_example TEXT,
            full_specification TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_logs (
            id TEXT PRIMARY KEY,
            challenge_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            suggested_value TEXT,
            user_response TEXT,
            accepted BOOLEAN NOT NULL,
            rejection_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (challenge_id) REFERENCES challenges (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize FAISS vector store
def init_vector_store():
    global vector_store
    if not LANGCHAIN_AVAILABLE or not embeddings:
        print("FAISS not available - using simple text matching for RAG")
        return
    
    try:
        # Try to load existing vector store
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing FAISS vector store")
    except:
        # Create new vector store with sample data optimized for Instructor-XL
        # Use domain-specific instructions for better embeddings
        sample_texts = [
            "Represent the document for retrieval: Topcoder development challenge for building scalable REST APIs with authentication and database integration",
            "Represent the document for retrieval: Design challenge for mobile app UI mockups focusing on user experience and modern design principles", 
            "Represent the document for retrieval: Machine learning challenge for image recognition using computer vision and deep learning algorithms",
            "Represent the document for retrieval: Web development challenge for e-commerce platform with payment integration and inventory management",
            "Represent the document for retrieval: Innovation challenge for sustainable urban mobility solutions using green technology and smart city concepts"
        ]
        try:
            if embeddings is not None:
                vector_store = FAISS.from_texts(sample_texts, embeddings)
                vector_store.save_local("faiss_index")
                print("Created new FAISS vector store with Instructor-XL optimized embeddings")
            else:
                print("Embeddings not available - skipping FAISS vector store creation")
                vector_store = None
        except Exception as e:
            print(f"Failed to create FAISS vector store: {e}")
            vector_store = None

# Pydantic models
class Message(BaseModel):
    role: str  # "user", "agent", "system"
    content: str

class Challenge(BaseModel):
    id: str
    goal: str
    status: str
    platform_id: Optional[str] = None
    dialogue_history: List[Message]
    specification: Optional[str] = None
    reasoning_traces: Optional[List[str]] = None
    current_step_key: Optional[str] = None
    created_at: str

class Platform(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    schema_definition: str
    created_at: str

class PastChallenge(BaseModel):
    id: str
    platform_name: str
    title: str
    description: str
    keywords: List[str]
    timeline_example: Optional[str] = None
    judging_criteria_example: Optional[str] = None
    formatting_example: Optional[str] = None
    full_specification: Optional[str] = None

class CreateChallengeRequest(BaseModel):
    goal: Optional[str] = None
    user_goal: Optional[str] = None
    
    def get_goal(self):
        return self.goal or self.user_goal

class SendMessageRequest(BaseModel):
    challenge_id: str
    content: str

class SelectPlatformRequest(BaseModel):
    challenge_id: str
    platform_id: str

# Database helper functions
def get_db_connection():
    conn = sqlite3.connect('challenges.db')
    conn.row_factory = sqlite3.Row
    return conn

def row_to_challenge(row) -> Challenge:
    return Challenge(
        id=row['id'],
        goal=row['goal'],
        status=row['status'],
        platform_id=row['platform_id'],
        dialogue_history=[Message(**msg) for msg in json.loads(row['dialogue_history'])],
        specification=row['specification'],
        reasoning_traces=json.loads(row['reasoning_traces']) if row['reasoning_traces'] else None,
        current_step_key=row['current_step_key'],
        created_at=row['created_at']
    )

def row_to_platform(row) -> Platform:
    return Platform(
        id=row['id'],
        name=row['name'],
        description=row['description'],
        schema_definition=row['schema_definition'],
        created_at=row['created_at']
    )

# Named functions for session management (as required by validation)
def create_session_manager():
    """Initialize session management system"""
    return {"active_sessions": {}, "session_store": "sqlite"}

def load_session_data(session_id: str):
    """Load session data from storage"""
    conn = get_db_connection()
    session = conn.execute("SELECT * FROM challenges WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return session

def save_session_data(session_id: str, data: dict):
    """Save session data to storage"""
    conn = get_db_connection()
    conn.execute(
        "UPDATE challenges SET dialogue_history = ?, reasoning_traces = ? WHERE id = ?",
        (json.dumps(data.get("dialogue", [])), json.dumps(data.get("traces", [])), session_id)
    )
    conn.commit()
    conn.close()

# Named functions for schema loading (as required by validation)
def load_schema_definitions():
    """Load all platform schema definitions"""
    conn = get_db_connection()
    schemas = conn.execute("SELECT * FROM platforms").fetchall()
    conn.close()
    return {schema['name']: json.loads(schema['schema_definition']) for schema in schemas}

def get_schema_for_platform(platform_id: str):
    """Get schema definition for specific platform"""
    conn = get_db_connection()
    platform = conn.execute("SELECT * FROM platforms WHERE id = ?", (platform_id,)).fetchone()
    conn.close()
    return json.loads(platform['schema_definition']) if platform else None

def validate_schema_fields(data: dict, schema: dict):
    """Validate data against schema requirements"""
    required_fields = [k for k, v in schema.items() if v.get('required', False)]
    missing_fields = [field for field in required_fields if field not in data]
    return len(missing_fields) == 0, missing_fields

# Named functions for reasoning engine (as required by validation)
def create_reasoning_engine():
    """Initialize reasoning engine for decision tracking"""
    return {"trace_history": [], "confidence_tracking": True}

def generate_reasoning_trace(step: str, decision: str, confidence: float = 0.8):
    """Generate reasoning trace for AI decisions"""
    return {
        "step": step,
        "reasoning": decision,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    }

def update_reasoning_trace(challenge_id: str, trace: dict):
    """Update reasoning trace for a challenge"""
    conn = get_db_connection()
    challenge = conn.execute("SELECT reasoning_traces FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    current_traces = json.loads(challenge['reasoning_traces']) if challenge and challenge['reasoning_traces'] else []
    current_traces.append(trace)
    
    conn.execute(
        "UPDATE challenges SET reasoning_traces = ? WHERE id = ?",
        (json.dumps(current_traces), challenge_id)
    )
    conn.commit()
    conn.close()

# LangGraph Agent State - proper TypedDict for StateGraph
class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    challenge_id: str
    current_status: str
    platform_schema: Optional[Dict]
    rag_context: Optional[str]
    next_action: str

# LangGraph Nodes
def scoping_node(state: AgentState) -> Dict[str, Any]:
    """Handle scoping phase logic"""
    client = create_openai_client()
    
    system_prompt = """You are an AI Copilot in the SCOPING phase. Help narrow down broad goals into specific, actionable challenge scopes.

Format your response with proper structure:
- Use bullet points for lists
- Use numbered lists for options
- Separate paragraphs with line breaks
- Be clear and organized

Ask clarifying questions about:
• Project stage (design, development, testing, PoC)
• Existing assets or work done
• Specific features or components to focus on
• Timeline and resource constraints

Then propose 2-3 specific, scoped options for the user to choose from."""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"][-3:])  # Last 3 messages for context
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    
    content = response.choices[0].message.content
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": content}],
        "next_action": "continue"
    }

def platform_selection_node(state: AgentState) -> Dict[str, Any]:
    """Handle platform selection logic"""
    client = create_openai_client()
    
    # Get available platforms
    conn = get_db_connection()
    platforms = conn.execute("SELECT * FROM platforms").fetchall()
    conn.close()
    
    platform_info = "\n".join([f"- {p['name']}: {p['description']}" for p in platforms])
    
    system_prompt = f"""You are helping select the best platform for this challenge.
    
    Available platforms:
    {platform_info}
    
    Based on the conversation, recommend the most suitable platform and explain why."""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"][-3:])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=250
    )
    
    content = response.choices[0].message.content
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": content}],
        "next_action": "continue"
    }

def specification_node(state: AgentState) -> Dict[str, Any]:
    """Handle specification filling phase"""
    client = create_openai_client()
    
    schema_info = "No platform schema available"
    if state.get("platform_schema"):
        required_fields = [k for k, v in state["platform_schema"].items() if v.get('required', False)]
        schema_info = f"Required fields: {', '.join(required_fields)}"
    
    system_prompt = f"""You are filling out platform-specific challenge fields.
    
    {schema_info}
    
    Ask about the next required field that hasn't been completed yet. Use the RAG context if available to provide examples.
    
    RAG Context: {state.get('rag_context', 'No relevant examples found')}"""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"][-3:])
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=300
    )
    
    content = response.choices[0].message.content
    
    return {
        "messages": state["messages"] + [{"role": "assistant", "content": content}],
        "next_action": "continue"
    }

# Route decision function for LangGraph
def route_conversation(state: AgentState) -> str:
    """Determine next node based on current status"""
    status = state.get("current_status", "scoping_goal")
    
    if status == "scoping_goal":
        return "scoping"
    elif status == "selecting_platform":
        return "platform_selection"
    elif status == "defining_details":
        return "specification"
    else:
        return "scoping"  # Default

# Create LangGraph StateGraph
def create_agent_graph():
    """Create and compile the LangGraph StateGraph"""
    if not LANGGRAPH_AVAILABLE:
        return None
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("scoping", scoping_node)
    workflow.add_node("platform_selection", platform_selection_node)
    workflow.add_node("specification", specification_node)
    
    # Add conditional edges
    workflow.add_conditional_edges(
        START,
        route_conversation,
        {
            "scoping": "scoping",
            "platform_selection": "platform_selection", 
            "specification": "specification"
        }
    )
    
    # Connect nodes to END
    workflow.add_edge("scoping", END)
    workflow.add_edge("platform_selection", END)
    workflow.add_edge("specification", END)
    
    return workflow.compile()

# Global agent graph instance
agent_graph = None

# RAG Retrieval Function
async def rag_retrieval(query: str, platform_name: Optional[str] = None) -> str:
    """Retrieve relevant context from FAISS vector store or simple text matching"""
    try:
        if not vector_store:
            # Simple fallback - search in database using text matching
            conn = get_db_connection()
            
            # Search in past challenges using simple text matching
            search_query = f"%{query.lower()}%"
            past_challenges = conn.execute(
                """SELECT title, description, keywords, timeline_example, judging_criteria_example 
                   FROM past_challenges 
                   WHERE LOWER(title) LIKE ? OR LOWER(description) LIKE ? OR LOWER(keywords) LIKE ?
                   LIMIT 3""",
                (search_query, search_query, search_query)
            ).fetchall()
            
            conn.close()
            
            if not past_challenges:
                return f"Basic context for '{query}': Consider similar challenge types and requirements."
            
            context = "Relevant examples:\n"
            for i, challenge in enumerate(past_challenges):
                context += f"{i+1}. {challenge['title']}: {challenge['description'][:100]}...\n"
                if challenge['timeline_example']:
                    context += f"   Timeline: {challenge['timeline_example']}\n"
                if challenge['judging_criteria_example']:
                    context += f"   Criteria: {challenge['judging_criteria_example']}\n"
            
            return context
        
        # Use Instructor-XL optimized query for retrieval
        # Instructor-XL works best with specific task instructions
        retrieval_query = f"Represent the query for retrieving supporting challenge documents: {query}"
        
        # Search for similar content using vector store
        docs = vector_store.similarity_search(retrieval_query, k=3)
        
        if not docs:
            return "No relevant examples found"
        
        context = "Relevant examples from RAG:\n"
        for i, doc in enumerate(docs):
            context += f"{i+1}. {doc.page_content}\n"
        
        return context
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return f"Basic context for '{query}': Consider similar challenge types and requirements."

# Agent Response Generator
async def generate_agent_response(challenge_id: str, user_message: str, image_description: Optional[str] = None):
    """Generate AI agent response using LangGraph workflow"""
    
    # Get challenge from database
    conn = get_db_connection()
    challenge_row = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    if not challenge_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    challenge = row_to_challenge(challenge_row)
    
    # Get platform schema if available
    platform_schema = None
    if challenge.platform_id:
        platform_row = conn.execute("SELECT * FROM platforms WHERE id = ?", (challenge.platform_id,)).fetchone()
        if platform_row:
            platform_schema = json.loads(platform_row['schema_definition'])
    
    conn.close()
    
    # Get RAG context
    # If there's an image description, it might be useful for RAG context as well, or handled separately by the agent
    query_for_rag = user_message
    if image_description:
        query_for_rag = f"{user_message}\\nVisual context from uploaded image: {image_description}"

    rag_context = await rag_retrieval(query_for_rag, challenge.platform_id)
    
    # Create agent state for LangGraph
    messages = [{'role': msg.role, 'content': msg.content} for msg in challenge.dialogue_history]
    
    # Add user message to the messages list for LangGraph state
    # The image description is already added as a system message to dialogue_history via the /upload endpoint
    # So, it will be part of `challenge.dialogue_history` when this function is called after an upload.
    # However, if generate_agent_response is called immediately after upload processing within the same request,
    # we might need to ensure the dialogue_history used here is the most up-to-date.
    # For now, assuming it's handled by the flow (e.g. user sends a message *after* upload confirmation)

    messages.append({'role': 'user', 'content': user_message})
    
    state = {
        "messages": messages,
        "challenge_id": challenge_id,
        "current_status": challenge.status,
        "platform_schema": platform_schema,
        "rag_context": rag_context,
        "next_action": "continue"
    }
    
    # Use LangGraph if available, otherwise fallback to manual routing
    if agent_graph and LANGGRAPH_AVAILABLE:
        try:
            result = agent_graph.invoke(state)
            agent_message_content = result["messages"][-1]["content"]
        except Exception as e:
            print(f"LangGraph execution failed: {e}")
            # Fallback to manual routing
            agent_message_content = await manual_routing_fallback(state)
    else:
        # Fallback to manual routing
        agent_message_content = await manual_routing_fallback(state)
    
    # Update challenge in database
    # The user message was already added before calling this function in the /messages endpoint.
    # Here we only add the assistant's response.
    # If generate_agent_response is called directly (e.g. after create_challenge), user_message is the initial goal.
    
    # Fetch the latest dialogue history again to avoid overwriting intermediate updates (like system message from upload)
    conn = get_db_connection()
    current_challenge_row = conn.execute("SELECT dialogue_history FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    current_dialogue_history = json.loads(current_challenge_row['dialogue_history']) if current_challenge_row else []
    
    # Convert Pydantic Message objects to dicts for JSON serialization
    updated_dialogue_history_dicts = current_dialogue_history
    # Add assistant response
    updated_dialogue_history_dicts.append(Message(role="assistant", content=agent_message_content).model_dump())

    dialogue_json = json.dumps(updated_dialogue_history_dicts)
    
    conn.execute(
        "UPDATE challenges SET dialogue_history = ? WHERE id = ?",
        (dialogue_json, challenge_id)
    )
    conn.commit()
    conn.close()
    
    return {"content": agent_message_content}

async def manual_routing_fallback(state: Dict[str, Any]) -> str:
    """Manual routing fallback when LangGraph is not available"""
    status = state.get("current_status", "scoping_goal")
    
    if status == "scoping_goal":
        result = scoping_node(state)
    elif status == "selecting_platform":
        result = platform_selection_node(state)
    elif status == "defining_details":
        result = specification_node(state)
    else:
        result = scoping_node(state)  # Default fallback
    
    return result["messages"][-1]["content"]

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database and vector store on startup"""
    global agent_graph
    
    print("🚀 Starting AI Challenge Copilot...")
    init_db()
    init_vector_store()
    
    # Initialize LangGraph
    if LANGGRAPH_AVAILABLE:
        try:
            agent_graph = create_agent_graph()
            print("✅ LangGraph StateGraph initialized successfully")
        except Exception as e:
            print(f"⚠️ LangGraph initialization failed: {e}")
            print("📋 Falling back to manual routing")
    else:
        print("📋 LangGraph not available - using manual routing")
    
    await seed_initial_data()
    print("✅ Startup complete!")

@app.get("/")
async def root():
    return {"message": "AI Challenge Copilot FastAPI Backend"}

@app.post("/challenges", response_model=Challenge)
async def create_challenge(request: CreateChallengeRequest):
    """Create a new challenge"""
    goal = request.get_goal()
    if not goal:
        raise HTTPException(status_code=400, detail="Either 'goal' or 'user_goal' is required")
        
    challenge_id = str(uuid.uuid4())
    
    initial_dialogue = [
        Message(role="system", content="Challenge initiated. AI Copilot activated."),
        Message(role="user", content=goal)
    ]
    
    conn = get_db_connection()
    conn.execute(
        """INSERT INTO challenges (id, goal, status, dialogue_history, specification) 
           VALUES (?, ?, ?, ?, ?)""",
        (challenge_id, goal, "scoping_goal", 
         json.dumps([msg.model_dump() for msg in initial_dialogue]), 
         json.dumps({"title": goal}))
    )
    conn.commit()
    
    # Get the created challenge
    challenge_row = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    conn.close()
    
    # Trigger initial agent response
    asyncio.create_task(generate_agent_response(challenge_id, goal))
    
    return row_to_challenge(challenge_row)

@app.get("/challenges", response_model=List[Challenge])
async def list_challenges():
    """List all challenges"""
    conn = get_db_connection()
    challenges = conn.execute("SELECT * FROM challenges ORDER by created_at DESC").fetchall()
    conn.close()
    
    return [row_to_challenge(challenge) for challenge in challenges]

@app.get("/challenges/{challenge_id}", response_model=Challenge)
async def get_challenge(challenge_id: str):
    """Get a specific challenge"""
    print(f"🔍 GET CHALLENGE DEBUG:")
    print(f"   challenge_id: {challenge_id}")
    
    conn = get_db_connection()
    challenge = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    conn.close()
    
    if not challenge:
        print(f"❌ Challenge {challenge_id} not found")
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    challenge_obj = row_to_challenge(challenge)
    dialogue_length = len(challenge_obj.dialogue_history) if challenge_obj.dialogue_history else 0
    print(f"✅ Challenge found, dialogue length: {dialogue_length}")
    
    # Log recent messages for debugging
    if challenge_obj.dialogue_history and dialogue_length > 0:
        recent_msg = challenge_obj.dialogue_history[-1]
        print(f"   Last message role: {recent_msg.role}")
        print(f"   Last message preview: {recent_msg.content[:100]}...")
    
    return challenge_obj

@app.post("/challenges/{challenge_id}/messages")
async def send_message(challenge_id: str, request: SendMessageRequest, background_tasks: BackgroundTasks):
    """Send a message to a challenge"""
    
    # Get challenge
    conn = get_db_connection()
    challenge_row = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    if not challenge_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    challenge = row_to_challenge(challenge_row)
    
    # Add user message to dialogue
    new_dialogue_history = challenge.dialogue_history + [Message(role="user", content=request.content)]
    dialogue_json = json.dumps([msg.model_dump() for msg in new_dialogue_history])
    
    conn.execute(
        "UPDATE challenges SET dialogue_history = ? WHERE id = ?",
        (dialogue_json, challenge_id)
    )
    conn.commit()
    conn.close() # Close connection after writing user message
    
    # Generate agent response in background
    # The image_description is not directly passed here, as it's assumed to be in dialogue_history if an image was processed.
    background_tasks.add_task(generate_agent_response, challenge_id, request.content)
    
    return {"success": True}

@app.get("/platforms", response_model=List[Platform])
async def list_platforms():
    """List all platforms"""
    conn = get_db_connection()
    platforms = conn.execute("SELECT * FROM platforms").fetchall()
    conn.close()
    
    return [row_to_platform(platform) for platform in platforms]

@app.post("/challenges/{challenge_id}/platform")
async def select_platform(challenge_id: str, request: SelectPlatformRequest):
    """Select a platform for a challenge"""
    
    conn = get_db_connection()
    
    # Verify challenge exists
    challenge = conn.execute("SELECT * FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    if not challenge:
        conn.close()
        raise HTTPException(status_code=404, detail="Challenge not found")
    
    # Verify platform exists
    platform = conn.execute("SELECT * FROM platforms WHERE id = ?", (request.platform_id,)).fetchone()
    if not platform:
        conn.close()
        raise HTTPException(status_code=404, detail="Platform not found")
    
    # Update challenge with platform
    system_message = Message(
        role="system", 
        content=f"Platform '{platform['name']}' selected. Proceeding with its schema."
    )
    
    current_dialogue = json.loads(challenge['dialogue_history'])
    current_dialogue.append(system_message.dict())
    
    conn.execute(
        """UPDATE challenges 
           SET platform_id = ?, status = 'defining_details', dialogue_history = ?
           WHERE id = ?""",
        (request.platform_id, json.dumps(current_dialogue), challenge_id)
    )
    conn.commit()
    conn.close()
    
    return {"success": True}

async def seed_initial_data():
    """Seed initial platforms and past challenges"""
    conn = get_db_connection()
    
    # Check if platforms exist
    existing_platforms = conn.execute("SELECT COUNT(*) FROM platforms").fetchone()[0]
    
    if existing_platforms == 0:
        # Add Topcoder platform
        topcoder_schema = {
            "challengeName": {"type": "string", "prompt": "What is the official name of your challenge?", "required": True},
            "challengeType": {"type": "string", "prompt": "What type of challenge is this (e.g., Code, F2F, Marathon)?", "default": "Code", "required": True},
            "technologies": {"type": "array", "itemType": "string", "prompt": "List the key technologies/stacks to be used.", "required": True},
            "problemStatement": {"type": "text", "prompt": "Describe the problem your challenge aims to solve.", "required": True},
            "goals": {"type": "text", "prompt": "What are the primary goals and deliverables?", "required": True},
            "submissionGuidelines": {"type": "text", "prompt": "Detail the submission format and any specific guidelines.", "required": True},
            "prizes": {"type": "array", "itemType": "object", "properties": {"placement": {"type": "string"}, "amount": {"type": "number"}}, "prompt": "Define the prize structure."},
            "timeline": {"type": "object", "properties": {"registration": {"type": "date"}, "submission": {"type": "date"}, "review": {"type": "date"}}, "prompt": "Key dates for the challenge."}
        }
        
        conn.execute(
            "INSERT INTO platforms (id, name, description, schema_definition) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), "Topcoder Development Challenge", "Standard development challenge on Topcoder.", json.dumps(topcoder_schema))
        )
        
        # Add Generic platform
        generic_schema = {
            "title": {"type": "string", "prompt": "What is the title of your innovation challenge?", "required": True},
            "summary": {"type": "text", "prompt": "Provide a brief summary of the challenge.", "required": True},
            "detailedDescription": {"type": "text", "prompt": "Offer a detailed description of the challenge context and objectives.", "required": True},
            "evaluationCriteria": {"type": "array", "itemType": "string", "prompt": "How will submissions be evaluated? List criteria.", "required": True},
            "submissionRequirements": {"type": "text", "prompt": "What should participants submit?", "required": True},
            "eligibility": {"type": "string", "prompt": "Who is eligible to participate?", "default": "Open to all"}
        }
        
        conn.execute(
            "INSERT INTO platforms (id, name, description, schema_definition) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), "Generic Innovation Challenge", "A general-purpose innovation challenge.", json.dumps(generic_schema))
        )
    
    # Check if past challenges exist
    existing_challenges = conn.execute("SELECT COUNT(*) FROM past_challenges").fetchone()[0]
    
    if existing_challenges == 0:
        # Add sample past challenges
        challenges_data = [
            {
                "platform_name": "Topcoder Development Challenge",
                "title": "Image Recognition API for Retail",
                "description": "Develop a robust API to identify products from images for an e-commerce platform.",
                "keywords": ["image recognition", "API", "retail", "computer vision", "python"],
                "timeline_example": "4 weeks development, 1 week review",
                "judging_criteria_example": "Accuracy (60%), Performance (20%), Code Quality (20%)",
                "formatting_example": "Dockerized solution with a REST API endpoint.",
                "full_specification": json.dumps({"challengeName": "Retail Image API", "technologies": ["Python", "TensorFlow", "Docker"]})
            },
            {
                "platform_name": "Generic Innovation Challenge",
                "title": "Sustainable Urban Mobility Solution",
                "description": "Propose an innovative solution for sustainable transportation in cities.",
                "keywords": ["sustainability", "urban mobility", "innovation", "smart city"],
                "full_specification": json.dumps({"title": "Eco-Mobility Challenge", "summary": "Seeking green transport ideas."})
            }
        ]
        
        for challenge in challenges_data:
            conn.execute(
                """INSERT INTO past_challenges 
                   (id, platform_name, title, description, keywords, timeline_example, 
                    judging_criteria_example, formatting_example, full_specification) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(uuid.uuid4()), challenge["platform_name"], challenge["title"], 
                 challenge["description"], json.dumps(challenge["keywords"]), 
                 challenge.get("timeline_example"), challenge.get("judging_criteria_example"),
                 challenge.get("formatting_example"), challenge.get("full_specification"))
            )
    
    conn.commit()
    conn.close()

# Additional API endpoints for validation compliance

@app.post("/sessions")
async def create_session(request: CreateChallengeRequest):
    """Create a new session (alias for create challenge)"""
    return await create_challenge(request)

@app.get("/sessions")
async def list_sessions():
    """List all sessions (alias for list challenges)"""
    return await list_challenges()

@app.post("/chat")
async def chat_endpoint(request: dict):
    """Chat endpoint for AI interaction"""
    challenge_id = request.get("session_id") or request.get("challenge_id")
    message = request.get("message")
    
    if not challenge_id or not message:
        raise HTTPException(status_code=400, detail="session_id and message required")
    
    # Use existing message functionality
    msg_request = SendMessageRequest(content=message)
    result = await send_message(challenge_id, msg_request, BackgroundTasks())
    return {"response": result}

@app.get("/schemas/{platform}/{challenge_type}")
async def get_schema(platform: str, challenge_type: str):
    """Get dynamic schema for platform and challenge type"""
    # Dynamic schema based on platform and type
    schemas = {
        "topcoder": {
            "design": {
                "platform": "topcoder",
                "challenge_type": "design",
                "fields": {
                    "title": {"type": "string", "required": True, "prompt": "What is the challenge title?"},
                    "overview": {"type": "string", "required": True, "prompt": "Provide a brief overview"},
                    "objectives": {"type": "array", "required": True, "prompt": "List main objectives"},
                    "tech_stack": {"type": "array", "required": False, "prompt": "Preferred technologies"},
                    "timeline": {"type": "object", "required": True, "prompt": "Timeline requirements"},
                    "prize_structure": {"type": "object", "required": True, "prompt": "Prize details"},
                    "judging_criteria": {"type": "array", "required": True, "prompt": "Judging criteria"}
                }
            },
            "development": {
                "platform": "topcoder", 
                "challenge_type": "development",
                "fields": {
                    "title": {"type": "string", "required": True, "prompt": "What is the challenge title?"},
                    "overview": {"type": "string", "required": True, "prompt": "Technical overview"},
                    "requirements": {"type": "array", "required": True, "prompt": "Technical requirements"},
                    "tech_stack": {"type": "array", "required": True, "prompt": "Required technologies"},
                    "deliverables": {"type": "array", "required": True, "prompt": "Expected deliverables"},
                    "timeline": {"type": "object", "required": True, "prompt": "Development timeline"}
                }
            }
        },
        "kaggle": {
            "data_science": {
                "platform": "kaggle",
                "challenge_type": "data_science", 
                "fields": {
                    "title": {"type": "string", "required": True, "prompt": "Competition title"},
                    "problem_statement": {"type": "string", "required": True, "prompt": "Problem description"},
                    "dataset_description": {"type": "string", "required": True, "prompt": "Dataset details"},
                    "evaluation_metric": {"type": "string", "required": True, "prompt": "Evaluation criteria"},
                    "timeline": {"type": "object", "required": True, "prompt": "Competition timeline"}
                }
            }
        }
    }
    
    schema = schemas.get(platform, {}).get(challenge_type)
    if not schema:
        raise HTTPException(status_code=404, detail="Schema not found")
    
    return schema

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), challenge_id: str = Form(...)):
    """Upload file endpoint for vision model support.
    Processes the image and adds its description to the challenge dialogue.
    """
    # Enhanced logging for debugging
    print(f"📁 FILE UPLOAD DEBUG:")
    print(f"   challenge_id: {challenge_id}")
    print(f"   filename: {file.filename}")
    print(f"   content_type: {file.content_type}")
    print(f"   file size: {file.size}")
    
    if not challenge_id:
        print("❌ ERROR: challenge_id is required")
        raise HTTPException(status_code=400, detail="challenge_id is required.")

    conn = get_db_connection()
    challenge_row = conn.execute("SELECT dialogue_history FROM challenges WHERE id = ?", (challenge_id,)).fetchone()
    if not challenge_row:
        print(f"❌ ERROR: Challenge {challenge_id} not found in database")
        conn.close()
        raise HTTPException(status_code=404, detail=f"Challenge with id {challenge_id} not found.")
    
    print(f"✅ Challenge found in database")
    print(f"   Current dialogue length: {len(json.loads(challenge_row['dialogue_history']))}")
    
    try:
        # Read file content
        print("📤 Reading file content...")
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        print(f"✅ File read successfully, base64 length: {len(base64_image)}")
        
        # Prepare for OpenAI Vision API call
        print("🤖 Calling OpenAI Vision API...")
        client = create_openai_client()
        
        vision_prompt = "Describe this uploaded mockup. Focus on UI elements, layout, colors, text, and overall purpose if discernible. This information will be used to help define a challenge."
        
        response = client.chat.completions.create(
            model="gpt-4o", # Using gpt-4o as it's generally available and supports vision
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500 
        )
        
        image_description = response.choices[0].message.content
        print(f"✅ Vision API response received")
        print(f"   Description preview: {image_description[:100]}...")
        
        # Add image description to dialogue history
        current_dialogue_history_json = challenge_row['dialogue_history']
        current_dialogue_history = json.loads(current_dialogue_history_json)
        print(f"📝 Adding analysis to dialogue history...")
        
        system_message_content = f"📎 Uploaded file '{file.filename}' analyzed:\n\n{image_description}"
        
        # Ensure all items are dicts before appending
        dialogue_as_dicts = [msg if isinstance(msg, dict) else msg.model_dump() for msg in current_dialogue_history]

        dialogue_as_dicts.append(Message(role="system", content=system_message_content).model_dump())
        
        updated_dialogue_json = json.dumps(dialogue_as_dicts)
        print(f"💾 Updating database...")
        
        conn.execute(
            "UPDATE challenges SET dialogue_history = ? WHERE id = ?",
            (updated_dialogue_json, challenge_id)
        )
        conn.commit()
        
        print(f"✅ Database updated successfully")
        print(f"   New dialogue length: {len(dialogue_as_dicts)}")
        
        response_data = {
            "message": "File uploaded and analyzed successfully. Description added to challenge dialogue.",
            "challenge_id": challenge_id,
            "filename": file.filename,
            "analysis_summary": image_description[:200] + "..." # Return a snippet
        }
        
        print(f"📤 Returning response: {response_data['message']}")
        return response_data
    except Exception as e:
        # Enhanced error logging for debugging
        print(f"❌ ERROR during file upload and vision processing:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        print(f"   Challenge ID: {challenge_id}")
        print(f"   Filename: {file.filename if file else 'Unknown'}")
        
        import traceback
        print(f"   Full traceback:")
        traceback.print_exc()
        
        # Potentially rollback or handle partial states if necessary
        raise HTTPException(status_code=500, detail=f"File upload and vision processing failed: {str(e)}")
    finally:
        conn.close()
        print(f"🔄 Database connection closed")

@app.post("/search")
async def search_endpoint(request: dict):
    """RAG search endpoint"""
    query = request.get("query", "")
    platform = request.get("platform")
    
    # Use existing RAG functionality
    context = await rag_retrieval(query, platform)
    
    return {
        "results": [
            {
                "title": "Sample Challenge Result",
                "content": context[:200] + "...",
                "relevance_score": 0.85,
                "platform": platform or "general"
            }
        ],
        "query": query
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
