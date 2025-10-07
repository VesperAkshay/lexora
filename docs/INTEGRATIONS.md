# Integration Guide

This guide shows how to integrate Lexora with popular frameworks and platforms.

## Table of Contents

- [FastAPI](#fastapi)
- [Streamlit](#streamlit)
- [Django](#django)
- [Flask](#flask)
- [LangChain](#langchain)
- [Gradio](#gradio)
- [Discord Bot](#discord-bot)
- [Slack Bot](#slack-bot)

## FastAPI

Build production-ready REST APIs with FastAPI and Lexora.

### Basic FastAPI Integration

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from lexora import RAGAgent
from typing import Optional, List
import asyncio

app = FastAPI(title="Lexora RAG API", version="1.0.0")

# Initialize agent (consider using dependency injection)
agent = RAGAgent()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    corpus_name: Optional[str] = None
    top_k: int = 5
    metadata_filters: Optional[dict] = None

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[dict]
    execution_time: float

class DocumentInput(BaseModel):
    content: str
    metadata: Optional[dict] = None

class CorpusCreate(BaseModel):
    name: str
    description: Optional[str] = None
    metadata: Optional[dict] = None

# Endpoints
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system."""
    try:
        response = await agent.query(
            query=request.query,
            corpus_name=request.corpus_name,
            top_k=request.top_k,
            metadata_filters=request.metadata_filters
        )
        
        return QueryResponse(
            answer=response.answer,
            confidence=response.confidence,
            sources=[{
                "content": s.content,
                "metadata": s.metadata,
                "score": s.score
            } for s in response.sources],
            execution_time=response.execution_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/corpus")
async def create_corpus(corpus: CorpusCreate):
    """Create a new corpus."""
    try:
        result = await agent.create_corpus(
            corpus_name=corpus.name,
            description=corpus.description,
            metadata=corpus.metadata
        )
        return {"message": f"Corpus '{corpus.name}' created successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/corpus")
async def list_corpora():
    """List all corpora."""
    try:
        corpora = await agent.list_corpora()
        return {"corpora": corpora}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/corpus/{corpus_name}/documents")
async def add_documents(corpus_name: str, documents: List[DocumentInput], background_tasks: BackgroundTasks):
    """Add documents to a corpus."""
    try:
        docs = [{"content": d.content, "metadata": d.metadata} for d in documents]
        result = await agent.add_documents(corpus_name, docs)
        return {
            "message": f"Added {result.documents_added} documents",
            "corpus_name": corpus_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/corpus/{corpus_name}")
async def delete_corpus(corpus_name: str):
    """Delete a corpus."""
    try:
        await agent.delete_corpus(corpus_name, confirm_deletion=corpus_name)
        return {"message": f"Corpus '{corpus_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "lexora-rag-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### FastAPI with Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, "your-secret-key", algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, user=Depends(verify_token)):
    """Protected query endpoint."""
    # Your query logic here
    pass
```

## Streamlit

Create interactive web applications with Streamlit.

### Complete Streamlit App

```python
import streamlit as st
import asyncio
from lexora import RAGAgent
import pandas as pd

# Page config
st.set_page_config(
    page_title="Lexora RAG Chat",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_agent():
    """Initialize and cache the RAG agent."""
    return RAGAgent()

def main():
    st.title("🤖 Lexora RAG Chat Interface")
    
    agent = get_agent()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Corpus selection
        corpora = asyncio.run(agent.list_corpora())
        corpus_names = [c["name"] for c in corpora]
        
        selected_corpus = st.selectbox(
            "Select Corpus",
            options=corpus_names if corpus_names else ["No corpora available"],
            index=0
        )
        
        # Query parameters
        st.subheader("Query Parameters")
        top_k = st.slider("Number of results", 1, 10, 5)
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.5)
        
        # Corpus management
        st.header("📚 Corpus Management")
        
        with st.expander("Create New Corpus"):
            new_corpus_name = st.text_input("Corpus Name")
            new_corpus_desc = st.text_area("Description")
            
            if st.button("Create Corpus"):
                if new_corpus_name:
                    try:
                        asyncio.run(agent.create_corpus(
                            corpus_name=new_corpus_name,
                            description=new_corpus_desc
                        ))
                        st.success(f"✅ Created corpus: {new_corpus_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with st.expander("Upload Documents"):
            uploaded_files = st.file_uploader(
                "Choose files",
                accept_multiple_files=True,
                type=['txt', 'md', 'pdf']
            )
            
            if uploaded_files and selected_corpus != "No corpora available":
                if st.button("Upload Documents"):
                    documents = []
                    for file in uploaded_files:
                        content = file.read().decode('utf-8')
                        documents.append({
                            "content": content,
                            "metadata": {"filename": file.name}
                        })
                    
                    try:
                        result = asyncio.run(agent.add_documents(selected_corpus, documents))
                        st.success(f"✅ Added {len(documents)} documents")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        # Statistics
        st.header("📊 Statistics")
        if selected_corpus != "No corpora available":
            info = asyncio.run(agent.get_corpus_info(selected_corpus))
            st.metric("Documents", info.get("document_count", 0))
            st.metric("Created", info.get("created_at", "N/A"))
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}** (Score: {source.get('score', 0):.3f})")
                        st.write(source.get('content', '')[:200] + "...")
                        if source.get('metadata'):
                            st.json(source['metadata'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        if selected_corpus == "No corpora available":
            st.error("Please create a corpus first!")
            return
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = asyncio.run(agent.query(
                        query=prompt,
                        corpus_name=selected_corpus,
                        top_k=top_k
                    ))
                    
                    # Display answer
                    st.markdown(response.answer)
                    
                    # Display confidence
                    st.progress(response.confidence)
                    st.caption(f"Confidence: {response.confidence:.2%}")
                    
                    # Store message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": [{
                            "content": s.content,
                            "score": s.score,
                            "metadata": s.metadata
                        } for s in response.sources]
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
```

## Django

Integrate Lexora with Django applications.

### Django Views

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views import View
import json
import asyncio
from lexora import RAGAgent
from asgiref.sync import sync_to_async

# Initialize agent (consider using Django cache)
agent = RAGAgent()

@csrf_exempt
@require_http_methods(["POST"])
def query_view(request):
    """Handle query requests."""
    try:
        data = json.loads(request.body)
        query = data.get('query')
        corpus_name = data.get('corpus_name')
        
        if not query:
            return JsonResponse({'error': 'Query is required'}, status=400)
        
        # Run async function
        response = asyncio.run(agent.query(query, corpus_name=corpus_name))
        
        return JsonResponse({
            'answer': response.answer,
            'confidence': response.confidence,
            'sources': [{
                'content': s.content,
                'metadata': s.metadata,
                'score': s.score
            } for s in response.sources]
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

class CorpusView(View):
    """Handle corpus operations."""
    
    @csrf_exempt
    def dispatch(self, *args, **kwargs):
        return super().dispatch(*args, **kwargs)
    
    def get(self, request):
        """List all corpora."""
        try:
            corpora = asyncio.run(agent.list_corpora())
            return JsonResponse({'corpora': corpora})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    def post(self, request):
        """Create a new corpus."""
        try:
            data = json.loads(request.body)
            corpus_name = data.get('name')
            description = data.get('description')
            
            result = asyncio.run(agent.create_corpus(corpus_name, description))
            return JsonResponse({'message': f'Corpus {corpus_name} created'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/query/', views.query_view, name='query'),
    path('api/corpus/', views.CorpusView.as_view(), name='corpus'),
]
```

### Django Management Command

```python
# management/commands/load_documents.py
from django.core.management.base import BaseCommand
from lexora import RAGAgent
import asyncio
import os

class Command(BaseCommand):
    help = 'Load documents into Lexora corpus'
    
    def add_arguments(self, parser):
        parser.add_argument('corpus_name', type=str)
        parser.add_argument('directory', type=str)
    
    def handle(self, *args, **options):
        corpus_name = options['corpus_name']
        directory = options['directory']
        
        agent = RAGAgent()
        documents = []
        
        # Read all files
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    documents.append({
                        'content': content,
                        'metadata': {'filename': filename}
                    })
        
        # Add to corpus
        result = asyncio.run(agent.add_documents(corpus_name, documents))
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully loaded {result.documents_added} documents')
        )
```

## Flask

Simple Flask integration for lightweight applications.

```python
from flask import Flask, request, jsonify
from lexora import RAGAgent
import asyncio

app = Flask(__name__)
agent = RAGAgent()

@app.route('/query', methods=['POST'])
def query():
    """Query endpoint."""
    data = request.get_json()
    query_text = data.get('query')
    corpus_name = data.get('corpus_name')
    
    if not query_text:
        return jsonify({'error': 'Query is required'}), 400
    
    try:
        response = asyncio.run(agent.query(query_text, corpus_name=corpus_name))
        return jsonify({
            'answer': response.answer,
            'confidence': response.confidence,
            'sources': [{
                'content': s.content,
                'score': s.score
            } for s in response.sources]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/corpus', methods=['POST'])
def create_corpus():
    """Create corpus endpoint."""
    data = request.get_json()
    corpus_name = data.get('name')
    description = data.get('description')
    
    try:
        asyncio.run(agent.create_corpus(corpus_name, description))
        return jsonify({'message': f'Corpus {corpus_name} created'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/corpus/<corpus_name>/documents', methods=['POST'])
def add_documents(corpus_name):
    """Add documents endpoint."""
    data = request.get_json()
    documents = data.get('documents', [])
    
    try:
        result = asyncio.run(agent.add_documents(corpus_name, documents))
        return jsonify({
            'message': f'Added {result.documents_added} documents'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## LangChain

Integrate Lexora as a LangChain retriever.

```python
from langchain.schema import BaseRetriever, Document
from lexora import RAGAgent
from typing import List
import asyncio

class LexoraRetriever(BaseRetriever):
    """LangChain retriever using Lexora."""
    
    agent: RAGAgent
    corpus_name: str
    top_k: int = 5
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents."""
        response = asyncio.run(
            self.agent.search_documents(
                corpus_name=self.corpus_name,
                query=query,
                top_k=self.top_k
            )
        )
        
        return [
            Document(
                page_content=result.content,
                metadata=result.metadata
            )
            for result in response
        ]
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async retrieve relevant documents."""
        response = await self.agent.search_documents(
            corpus_name=self.corpus_name,
            query=query,
            top_k=self.top_k
        )
        
        return [
            Document(
                page_content=result.content,
                metadata=result.metadata
            )
            for result in response
        ]

# Usage with LangChain
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

agent = RAGAgent()
retriever = LexoraRetriever(agent=agent, corpus_name="my_corpus", top_k=5)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever
)

result = qa_chain.run("What is machine learning?")
print(result)
```

## Gradio

Create interactive demos with Gradio.

```python
import gradio as gr
from lexora import RAGAgent
import asyncio

agent = RAGAgent()

async def query_rag(message, history, corpus_name, top_k):
    """Query the RAG system."""
    try:
        response = await agent.query(
            query=message,
            corpus_name=corpus_name,
            top_k=top_k
        )
        
        # Format response with sources
        answer = f"{response.answer}\n\n**Confidence:** {response.confidence:.2%}\n\n"
        answer += "**Sources:**\n"
        for i, source in enumerate(response.sources, 1):
            answer += f"{i}. {source.content[:100]}...\n"
        
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Lexora RAG Chat") as demo:
    gr.Markdown("# 🤖 Lexora RAG Chat Interface")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(label="Your Question", placeholder="Ask a question...")
            clear = gr.Button("Clear")
        
        with gr.Column(scale=1):
            corpus_name = gr.Textbox(label="Corpus Name", value="my_corpus")
            top_k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K Results")
    
    def respond(message, chat_history, corpus, k):
        response = asyncio.run(query_rag(message, chat_history, corpus, k))
        chat_history.append((message, response))
        return "", chat_history
    
    msg.submit(respond, [msg, chatbot, corpus_name, top_k], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()
```

## Discord Bot

Build a Discord bot with Lexora.

```python
import discord
from discord.ext import commands
from lexora import RAGAgent
import asyncio

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
agent = RAGAgent()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='ask')
async def ask_question(ctx, *, question: str):
    """Ask a question to the RAG system."""
    async with ctx.typing():
        try:
            response = await agent.query(question, corpus_name="discord_kb")
            
            embed = discord.Embed(
                title="Answer",
                description=response.answer,
                color=discord.Color.blue()
            )
            embed.add_field(name="Confidence", value=f"{response.confidence:.2%}")
            embed.set_footer(text=f"Sources: {len(response.sources)}")
            
            await ctx.send(embed=embed)
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")

@bot.command(name='addcorpus')
async def add_corpus(ctx, name: str, description: str = ""):
    """Create a new corpus."""
    try:
        await agent.create_corpus(corpus_name=name, description=description)
        await ctx.send(f"✅ Created corpus: {name}")
    except Exception as e:
        await ctx.send(f"❌ Error: {str(e)}")

bot.run('YOUR_BOT_TOKEN')
```

## Slack Bot

Create a Slack bot integration.

```python
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from lexora import RAGAgent

app = AsyncApp(token="YOUR_BOT_TOKEN")
agent = RAGAgent()

@app.message("ask")
async def handle_ask(message, say):
    """Handle ask command."""
    question = message['text'].replace('ask', '').strip()
    
    try:
        response = await agent.query(question, corpus_name="slack_kb")
        
        await say({
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Answer:*\n{response.answer}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Confidence: {response.confidence:.2%} | Sources: {len(response.sources)}"
                        }
                    ]
                }
            ]
        })
    except Exception as e:
        await say(f"Error: {str(e)}")

async def main():
    handler = AsyncSocketModeHandler(app, "YOUR_APP_TOKEN")
    await handler.start_async()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

For more integration examples, visit our [Examples Repository](https://github.com/your-org/lexora-examples).
