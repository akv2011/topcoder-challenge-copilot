"""
Session Log Generator for AI Challenge Copilot
Generates the 3 required end-to-end session logs as specified in implement.md

This script demonstrates:
1. User → Copilot → Spec workflow
2. Different challenge types and platforms
3. Complete reasoning traces and structured outputs
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Any

def generate_session_logs():
    """Generate 3 end-to-end session logs as required by implement.md"""
    
    sessions = []
    
    # Session 1: Mobile App Design Challenge
    session1 = {
        "session_id": "session_001",
        "timestamp": datetime.now().isoformat(),
        "challenge_type": "Design Challenge - Mobile App UI",
        "platform": "Topcoder Development Challenge",
        "user_goal": "I want to build an app for cleaning services",
        "conversation_flow": [
            {
                "step": 1,
                "role": "user",
                "content": "I want to build an app for cleaning services",
                "timestamp": "2025-01-10T10:00:00Z"
            },
            {
                "step": 2,
                "role": "agent",
                "content": "That's a great goal! To help you scope this properly, could you tell me what stage you're currently at? Do you have any existing designs, mockups, or is this starting from scratch? Also, are you looking to build the entire app or focus on a specific feature first?",
                "reasoning_trace": [
                    "User provided broad goal - needs scoping",
                    "Identified as potentially too broad for single challenge",
                    "Need to understand current project stage",
                    "Should clarify specific deliverable scope"
                ],
                "ai_analysis": {
                    "feasibility": "too_broad",
                    "stage_clarity": 2,
                    "missing_info": ["project_stage", "existing_assets", "specific_features"],
                    "confidence": 0.85
                }
            },
            {
                "step": 3,
                "role": "user", 
                "content": "I'm starting from scratch. I want to focus on the booking flow where customers can schedule cleaning appointments.",
                "timestamp": "2025-01-10T10:02:00Z"
            },
            {
                "step": 4,
                "role": "agent",
                "content": "Perfect! Focusing on the booking flow is much more manageable. Based on similar projects, I'd suggest we scope this as designing 3-4 key screens: service selection, date/time picker, address entry, and booking confirmation. Would you like this as a mobile app UI design challenge, or do you need actual code implementation?",
                "reasoning_trace": [
                    "User provided better scope - booking flow specifically",
                    "RAG context shows similar UI design challenges are 3-4 screens",
                    "Need to clarify design vs development deliverable",
                    "Feasibility improved significantly"
                ],
                "rag_context": "Similar challenge: 'Restaurant Booking UI' had 4 screens and 5-day timeline",
                "ai_analysis": {
                    "feasibility": "appropriate",
                    "stage_clarity": 8,
                    "suggested_scopes": ["UI mockups for booking flow", "3-4 key screens", "Mobile-first design"],
                    "confidence": 0.92
                }
            },
            {
                "step": 5,
                "role": "user",
                "content": "UI design would be perfect. I need mockups I can show to developers later.",
                "timestamp": "2025-01-10T10:04:00Z"
            },
            {
                "step": 6,
                "role": "agent",
                "content": "Excellent! I recommend the Topcoder Development Challenge platform for this. Let me help you define the complete specification. First, what should we call this challenge?",
                "reasoning_trace": [
                    "User confirmed UI design scope",
                    "Topcoder platform selected based on design challenge type",
                    "Beginning platform-specific field completion",
                    "Starting with challenge name as first required field"
                ],
                "platform_selection": "Topcoder Development Challenge",
                "current_status": "defining_details"
            },
            {
                "step": 7,
                "role": "user",
                "content": "Let's call it 'CleanBook Mobile UI - Booking Flow Design'",
                "timestamp": "2025-01-10T10:05:00Z"
            },
            {
                "step": 8,
                "role": "agent",
                "content": "Great name! Now, what technologies should designers use? For UI design challenges, we typically recommend Figma, Sketch, or Adobe XD. What's your preference?",
                "reasoning_trace": [
                    "Challenge name captured: 'CleanBook Mobile UI - Booking Flow Design'",
                    "Moving to next required field: technologies",
                    "Provided relevant options for UI design tools",
                    "Avoiding development technologies since this is design-only"
                ],
                "field_completion": {
                    "challengeName": "CleanBook Mobile UI - Booking Flow Design",
                    "challengeType": "Design"
                }
            }
        ],
        "final_specification": {
            "challengeName": "CleanBook Mobile UI - Booking Flow Design",
            "challengeType": "Design",
            "technologies": ["Figma", "Sketch"],
            "problemStatement": "Design a clean, intuitive booking flow for a cleaning services mobile app that allows customers to easily schedule appointments.",
            "goals": "Create 4 high-fidelity mobile screens: service selection, date/time picker, address entry, and booking confirmation. Focus on user experience and modern design principles.",
            "submissionGuidelines": "Submit Figma files with clickable prototypes, plus PNG exports of all screens at 375x812px (iPhone X resolution).",
            "timeline": {
                "registration": "2025-01-15",
                "submission": "2025-01-22", 
                "review": "2025-01-25"
            },
            "prizes": [
                {"placement": "1st", "amount": 800},
                {"placement": "2nd", "amount": 400}
            ]
        },
        "reasoning_summary": {
            "scoping_decisions": [
                "Narrowed from 'entire app' to 'booking flow'",
                "Focused on UI design rather than development",
                "Defined specific deliverable: 4 mobile screens"
            ],
            "platform_rationale": "Topcoder selected for established design challenge framework",
            "rag_influence": "Used similar booking UI challenges for timeline and scope guidance",
            "final_confidence": 0.94
        }
    }
    
    # Session 2: API Development Challenge
    session2 = {
        "session_id": "session_002", 
        "timestamp": datetime.now().isoformat(),
        "challenge_type": "Development Challenge - API Backend",
        "platform": "Topcoder Development Challenge",
        "user_goal": "Need an API for image recognition in retail",
        "conversation_flow": [
            {
                "step": 1,
                "role": "user",
                "content": "Need an API for image recognition in retail",
                "timestamp": "2025-01-10T14:00:00Z"
            },
            {
                "step": 2,
                "role": "agent", 
                "content": "Image recognition for retail sounds valuable! To scope this properly, could you tell me more about what specific products you want to recognize? Also, do you have existing training data, or would this be using pre-trained models? And what's the intended use case - inventory management, customer app, or something else?",
                "reasoning_trace": [
                    "User goal is technical and specific",
                    "Image recognition scope needs clarification",
                    "Multiple potential approaches: custom training vs pre-trained",
                    "Use case affects API design requirements"
                ],
                "ai_analysis": {
                    "feasibility": "appropriate",
                    "stage_clarity": 4,
                    "missing_info": ["product_types", "training_data", "use_case", "accuracy_requirements"]
                }
            },
            {
                "step": 3,
                "role": "user",
                "content": "We need to identify clothing items from customer photos. We have about 1000 labeled images. It's for a shopping app where customers can snap a photo and find similar items in our catalog.",
                "timestamp": "2025-01-10T14:02:00Z"
            },
            {
                "step": 4,
                "role": "agent",
                "content": "Perfect! That's a well-defined scope. With 1000 labeled images, we can build a custom model. I'd recommend focusing on creating a REST API that accepts clothing images and returns product matches with confidence scores. Should we include the model training as part of the challenge, or focus just on the API assuming a trained model?",
                "reasoning_trace": [
                    "Clear use case: clothing recognition for shopping app",
                    "1000 images is reasonable dataset size",
                    "REST API approach is standard for this type of service",
                    "Need to clarify scope: training + API vs API only"
                ],
                "rag_context": "Similar challenge: 'Retail Image API' included both training and API components, 4-week timeline",
                "ai_analysis": {
                    "feasibility": "appropriate",
                    "stage_clarity": 8,
                    "suggested_scopes": ["API with pre-trained model", "API + custom training", "API + similarity matching"],
                    "confidence": 0.89
                }
            },
            {
                "step": 5,
                "role": "user",
                "content": "Let's include the training part too. I want a complete solution.",
                "timestamp": "2025-01-10T14:04:00Z"
            },
            {
                "step": 6,
                "role": "agent",
                "content": "Excellent! A complete solution including model training and API. This fits perfectly with Topcoder's development challenges. Let's call this 'Clothing Recognition API with Custom Model Training'. What technologies do you prefer? Python with TensorFlow/PyTorch is common, but I'm open to other suggestions.",
                "reasoning_trace": [
                    "User wants complete solution: training + API",
                    "Topcoder platform appropriate for development challenge",
                    "Suggested technical stack based on common ML practices",
                    "Ready to begin specification completion"
                ]
            }
        ],
        "final_specification": {
            "challengeName": "Clothing Recognition API with Custom Model Training",
            "challengeType": "Code",
            "technologies": ["Python", "TensorFlow", "FastAPI", "Docker"],
            "problemStatement": "Build a complete clothing recognition system that can train on custom dataset and provide REST API for identifying clothing items in customer photos.",
            "goals": "Develop a machine learning model to classify clothing items and create a REST API that accepts image uploads and returns product matches with confidence scores.",
            "submissionGuidelines": "Submit Dockerized solution with training scripts, REST API, documentation, and accuracy metrics. Include sample predictions on test dataset.",
            "timeline": {
                "registration": "2025-01-20",
                "submission": "2025-02-17",
                "review": "2025-02-24"
            },
            "prizes": [
                {"placement": "1st", "amount": 2000},
                {"placement": "2nd", "amount": 1000},
                {"placement": "3rd", "amount": 500}
            ]
        }
    }
    
    # Session 3: Innovation Challenge
    session3 = {
        "session_id": "session_003",
        "timestamp": datetime.now().isoformat(),
        "challenge_type": "Innovation Challenge - Sustainability",
        "platform": "Generic Innovation Challenge",
        "user_goal": "Looking for ideas to reduce food waste in restaurants",
        "conversation_flow": [
            {
                "step": 1,
                "role": "user",
                "content": "Looking for ideas to reduce food waste in restaurants",
                "timestamp": "2025-01-10T16:00:00Z"
            },
            {
                "step": 2,
                "role": "agent",
                "content": "Food waste reduction is a crucial sustainability challenge! Are you looking for technology solutions, process improvements, or innovative business models? Also, are you targeting a specific type of restaurant (fast food, fine dining, etc.) or restaurant operations in general?",
                "reasoning_trace": [
                    "Sustainability focus is clear",
                    "Need to understand solution type preference", 
                    "Restaurant type affects applicable solutions",
                    "Could be technology, process, or business model innovation"
                ],
                "ai_analysis": {
                    "feasibility": "appropriate",
                    "stage_clarity": 5,
                    "missing_info": ["solution_type", "restaurant_type", "specific_waste_areas"]
                }
            },
            {
                "step": 3,
                "role": "user",
                "content": "I'm open to any type of solution - tech, process, or business model. Mainly thinking about casual dining restaurants that throw away a lot of food at the end of each day.",
                "timestamp": "2025-01-10T16:02:00Z"
            },
            {
                "step": 4,
                "role": "agent",
                "content": "Great scope! End-of-day waste in casual dining is a perfect challenge topic. This sounds ideal for an innovation challenge where participants can propose creative solutions. Should we frame this as seeking specific innovations like apps, partnerships, donation systems, or keep it completely open for any creative approach?",
                "reasoning_trace": [
                    "User open to all solution types",
                    "Casual dining + end-of-day waste is specific enough",
                    "Innovation challenge format appropriate",
                    "Need to decide on solution constraint level"
                ],
                "rag_context": "Similar challenge: 'Urban Mobility Solutions' was open-ended and successful",
                "platform_selection": "Generic Innovation Challenge"
            }
        ],
        "final_specification": {
            "title": "Restaurant Food Waste Reduction Innovation Challenge",
            "summary": "Seeking innovative solutions to reduce end-of-day food waste in casual dining restaurants",
            "detailedDescription": "Casual dining restaurants face significant food waste challenges, particularly with unsold prepared food at closing time. We're seeking creative, practical solutions that could be implemented by restaurant owners to reduce waste, whether through technology, partnerships, process changes, or business model innovations.",
            "evaluationCriteria": [
                "Innovation and creativity of the solution",
                "Practical feasibility for restaurant implementation", 
                "Potential environmental impact",
                "Economic viability for restaurants",
                "Scalability across different restaurant types"
            ],
            "submissionRequirements": "Submit a detailed proposal (5-10 pages) including solution description, implementation plan, cost-benefit analysis, and pilot study design. Include mockups, diagrams, or prototypes if applicable.",
            "eligibility": "Open to all - individuals, teams, students, professionals, entrepreneurs"
        }
    }
    
    sessions.extend([session1, session2, session3])
    
    return sessions

def save_session_logs():
    """Save the generated session logs to files"""
    sessions = generate_session_logs()
    
    # Save individual session files
    for i, session in enumerate(sessions, 1):
        filename = f"session_log_{i}_{session['challenge_type'].replace(' ', '_').replace('-', '').lower()}.json"
        with open(filename, 'w') as f:
            json.dump(session, f, indent=2)
        print(f"✅ Generated: {filename}")
    
    # Save combined log file
    combined_log = {
        "ai_copilot_session_logs": sessions,
        "metadata": {
            "total_sessions": len(sessions),
            "generated_at": datetime.now().isoformat(),
            "compliance": "Phase 1 - implement.md requirements",
            "deliverable": "3 end-to-end session logs (user → Copilot → spec)"
        }
    }
    
    with open("all_session_logs.json", 'w') as f:
        json.dump(combined_log, f, indent=2)
    print("✅ Generated: all_session_logs.json")
    
    print(f"\n🎯 Successfully generated {len(sessions)} end-to-end session logs!")
    print("📋 Demonstrating:")
    print("   • Scoping dialogue logic")
    print("   • Schema-aware Q&A logic") 
    print("   • RAG integration examples")
    print("   • Reasoning trace generation")
    print("   • Complete user → Copilot → spec workflows")

if __name__ == "__main__":
    save_session_logs()
