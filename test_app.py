#!/usr/bin/env python3
"""
Test script to verify all dependencies work correctly
"""

def test_imports():
    """Test all critical imports"""
    try:
        print("🔧 Testing FastAPI imports...")
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        print("✅ FastAPI imports successful")
        
        print("🔧 Testing OpenAI imports...")
        import openai
        print("✅ OpenAI imports successful")
        
        print("🔧 Testing LangChain imports...")
        import langchain
        from langchain_community.vectorstores import FAISS
        try:
            from langchain_openai import OpenAIEmbeddings
            print("✅ LangChain OpenAI embeddings available")
        except ImportError:
            from langchain.embeddings import OpenAIEmbeddings
            print("✅ LangChain legacy embeddings available")
        print("✅ LangChain imports successful")
        
        print("🔧 Testing LangGraph imports...")
        from langgraph.graph import StateGraph, START, END
        from typing_extensions import TypedDict
        print("✅ LangGraph imports successful")
        
        print("🔧 Testing FAISS imports...")
        import faiss
        print("✅ FAISS imports successful")
        
        print("🔧 Testing Sentence Transformers...")
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imports successful")
        
        print("🔧 Testing other dependencies...")
        import requests
        from python_multipart import parse_options_header
        print("✅ All other dependencies successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    try:
        print("🔧 Testing FastAPI app creation...")
        from fastapi import FastAPI
        app = FastAPI()
        print("✅ FastAPI app creation successful")
        
        print("🔧 Testing LangGraph StateGraph...")
        from langgraph.graph import StateGraph, START, END
        from typing_extensions import TypedDict
        
        class TestState(TypedDict):
            messages: list
        
        workflow = StateGraph(TestState)
        print("✅ LangGraph StateGraph creation successful")
        
        print("🔧 Testing FAISS vector store...")
        import faiss
        import numpy as np
        
        # Create a simple test index
        dimension = 128
        index = faiss.IndexFlatL2(dimension)
        vectors = np.random.random((10, dimension)).astype('float32')
        index.add(vectors)
        print(f"✅ FAISS index created with {index.ntotal} vectors")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting dependency tests...")
    
    imports_ok = test_imports()
    functionality_ok = test_basic_functionality()
    
    if imports_ok and functionality_ok:
        print("\n🎉 All tests passed! Dependencies are working correctly.")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
