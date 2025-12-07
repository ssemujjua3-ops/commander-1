#!/usr/bin/env python3
"""
🚀 COMMANDER AI - Bot Creation, Deployment & Chat Interface
Create bots → Deploy → Chat with them → Manage all in one place
"""

import os
import uuid
import time
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

# FastAPI
from fastapi import FastAPI, HTTPException, Header, Body, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# ==================== CONFIGURATION ====================
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-" + str(uuid.uuid4()))
CREATOR_EMAIL = os.environ.get("CREATOR_EMAIL", "ssemujjua3@gmail.com")
CREATOR_PASSWORD = os.environ.get("CREATOR_PASSWORD", "ChangeMe123!")
CREATOR_API_KEY = os.environ.get("CREATOR_API_KEY", "creator-" + str(uuid.uuid4())[:16])
OVERRIDE_TOKEN = os.environ.get("OVERRIDE_TOKEN", "override-" + str(uuid.uuid4())[:16])
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
PORT = int(os.environ.get("PORT", 8000))

# ==================== APP INIT ====================
app = FastAPI(
    title="🤖 Commander AI - Chat & Bot Factory",
    description="Create bots, deploy them, and chat with them in real-time",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for assets
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================== DATABASE ====================
class Database:
    def __init__(self):
        self.users = {}
        self.bots = {}
        self.chats = {}  # chat_id -> messages
        self.deployments = {}
        self.conversations = {}  # user_id -> conversation history
        
        # Create creator user
        self.users[CREATOR_EMAIL] = {
            "id": str(uuid.uuid4()),
            "email": CREATOR_EMAIL,
            "password": CREATOR_PASSWORD,
            "api_key": CREATOR_API_KEY,
            "is_admin": True,
            "created_at": datetime.now().isoformat(),
            "chat_history": []
        }
        
        # Create default bots
        self.create_default_bots()
    
    def create_default_bots(self):
        """Create initial example bots"""
        default_bots = [
            {
                "id": "assistant-001",
                "name": "AI Assistant",
                "description": "General purpose AI assistant for answering questions",
                "skills": ["chat", "analysis", "explanation"],
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "status": "online",
                "owner": CREATOR_EMAIL,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "is_public": True,
                "avatar": "🤖"
            },
            {
                "id": "coder-001",
                "name": "Code Expert",
                "description": "Specialized in programming and code generation",
                "skills": ["python", "javascript", "code-review", "debugging"],
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "status": "online",
                "owner": CREATOR_EMAIL,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "is_public": True,
                "avatar": "💻"
            },
            {
                "id": "analyst-001",
                "name": "Data Analyst",
                "description": "Analyzes data and provides insights",
                "skills": ["analysis", "statistics", "visualization", "reports"],
                "model": "gpt-3.5-turbo",
                "temperature": 0.5,
                "status": "online",
                "owner": CREATOR_EMAIL,
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
                "is_public": True,
                "avatar": "📊"
            }
        ]
        
        for bot in default_bots:
            self.bots[bot["id"]] = bot
    
    def add_message(self, user_id: str, bot_id: str, role: str, content: str):
        """Add message to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = {}
        
        if bot_id not in self.conversations[user_id]:
            self.conversations[user_id][bot_id] = []
        
        message = {
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "bot_id": bot_id
        }
        
        self.conversations[user_id][bot_id].append(message)
        
        # Keep last 50 messages per conversation
        if len(self.conversations[user_id][bot_id]) > 50:
            self.conversations[user_id][bot_id] = self.conversations[user_id][bot_id][-50:]
        
        return message

db = Database()

# ==================== WEBSOCKET MANAGER ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_bot_map: Dict[str, str] = {}  # user_id -> bot_id
    
    async def connect(self, websocket: WebSocket, user_id: str, bot_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_bot_map[user_id] = bot_id
        print(f"✅ WebSocket connected: {user_id} -> {bot_id}")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_bot_map:
            del self.user_bot_map[user_id]
        print(f"❌ WebSocket disconnected: {user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

# ==================== OPENAI SERVICE (FIXED FOR DEPLOYMENT) ====================
class OpenAIService:
    def __init__(self):
        self.api_key = OPENAI_API_KEY
        self.enabled = bool(self.api_key and self.api_key.strip())
        
        if self.enabled:
            try:
                # Updated for OpenAI v1.x
                import openai
                openai.api_key = self.api_key
                self.client = openai.AsyncOpenAI(api_key=self.api_key)
                print("🧠 OpenAI: ✅ ENABLED")
            except ImportError:
                print("⚠️ OpenAI package not installed")
                self.enabled = False
                self.client = None
            except Exception as e:
                print(f"⚠️ OpenAI init error: {e}")
                self.enabled = False
                self.client = None
        else:
            print("🧠 OpenAI: ❌ DISABLED (no API key)")
            self.client = None
    
    async def chat_completion(self, messages: List[Dict], bot_config: Dict = None) -> str:
        """Chat with OpenAI API"""
        if not self.enabled or not self.client:
            return self._fallback_response(messages)
        
        try:
            # Use bot-specific configuration
            model = bot_config.get("model", "gpt-3.5-turbo") if bot_config else "gpt-3.5-turbo"
            temperature = bot_config.get("temperature", 0.7) if bot_config else 0.7
            
            # Updated for OpenAI v1.x
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")
            return self._fallback_response(messages)
    
    def _fallback_response(self, messages: List[Dict]) -> str:
        """Fallback when OpenAI is unavailable"""
        last_message = messages[-1]["content"] if messages else "Hello"
        
        responses = [
            f"I understand you said: '{last_message[:50]}...' This is a fallback response.",
            "I'm currently using fallback mode. To enable AI responses, add your OpenAI API key in Render environment variables.",
            f"Received your message about: {last_message[:30]}...",
            "Hello! I'm your AI assistant in fallback mode. Add OpenAI key for full functionality.",
            "I can help you with various tasks. Currently in basic mode - add API key for AI features."
        ]
        
        import random
        return random.choice(responses)

openai_service = OpenAIService()

# ==================== PYDANTIC MODELS ====================
class MessageCreate(BaseModel):
    content: str
    bot_id: str

class BotCreate(BaseModel):
    name: str
    description: str
    skills: List[str] = ["general"]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    is_public: bool = True
    avatar: str = "🤖"

class ChatRequest(BaseModel):
    message: str
    bot_id: str
    conversation_id: Optional[str] = None

# ==================== AUTHENTICATION ====================
def get_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    # Check creator API key
    if x_api_key == CREATOR_API_KEY:
        return {"user_id": CREATOR_EMAIL, "is_admin": True}
    
    # Check user API keys
    for email, user in db.users.items():
        if user.get("api_key") == x_api_key:
            return {"user_id": email, "is_admin": user.get("is_admin", False)}
    
    raise HTTPException(status_code=401, detail="Invalid API key")

# ==================== HEALTH ENDPOINTS ====================
@app.get("/")
async def root():
    return {
        "service": "🤖 Commander AI Chat",
        "version": "3.0.0",
        "status": "online",
        "features": ["chat", "bot_creation", "deployment", "websockets"],
        "openai": "enabled" if openai_service.enabled else "disabled",
        "endpoints": {
            "chat": "/chat",
            "chat_ui": "/chat-ui",
            "bots": "/api/bots",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "bots_online": len([b for b in db.bots.values() if b["status"] == "online"]),
        "active_connections": len(manager.active_connections),
        "openai": openai_service.enabled
    }

# ==================== CHAT ENDPOINTS ====================
@app.websocket("/ws/chat/{user_id}/{bot_id}")
async def websocket_chat(websocket: WebSocket, user_id: str, bot_id: str):
    """WebSocket for real-time chat"""
    await manager.connect(websocket, user_id, bot_id)
    
    try:
        # Send welcome message
        bot = db.bots.get(bot_id, {})
        welcome_msg = {
            "type": "system",
            "content": f"Connected to {bot.get('name', 'Bot')}. Start chatting!",
            "bot_id": bot_id,
            "timestamp": datetime.now().isoformat()
        }
        await websocket.send_json(welcome_msg)
        
        # Store initial user info
        user_data = {"user_id": user_id, "bot_id": bot_id}
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type", "message")
            
            if message_type == "message":
                user_message = data.get("content", "")
                
                if user_message:
                    # Save user message
                    db.add_message(user_id, bot_id, "user", user_message)
                    
                    # Send typing indicator
                    typing_msg = {
                        "type": "typing",
                        "bot_id": bot_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(typing_msg)
                    
                    # Get conversation history
                    history = db.conversations.get(user_id, {}).get(bot_id, [])
                    chat_messages = [
                        {"role": "system", "content": f"You are {bot.get('name')}. {bot.get('description')}"}
                    ]
                    
                    # Add last 10 messages for context
                    for msg in history[-10:]:
                        chat_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    # Add current message
                    chat_messages.append({"role": "user", "content": user_message})
                    
                    # Get AI response
                    ai_response = await openai_service.chat_completion(chat_messages, bot)
                    
                    # Save AI response
                    db.add_message(user_id, bot_id, "assistant", ai_response)
                    
                    # Update bot message count
                    if bot_id in db.bots:
                        db.bots[bot_id]["message_count"] += 1
                    
                    # Send response
                    response_msg = {
                        "type": "message",
                        "content": ai_response,
                        "sender": "bot",
                        "bot_id": bot_id,
                        "bot_name": bot.get("name", "Bot"),
                        "avatar": bot.get("avatar", "🤖"),
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send_json(response_msg)
            
            elif message_type == "typing":
                # Echo typing status
                typing_msg = {
                    "type": "typing",
                    "user_id": user_id,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_json(typing_msg)
                
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(user_id)

@app.post("/api/chat")
async def api_chat(request: ChatRequest, auth: dict = Depends(get_api_key)):
    """HTTP API for chat (for non-WebSocket clients)"""
    user_id = auth["user_id"]
    bot_id = request.bot_id
    
    if bot_id not in db.bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = db.bots[bot_id]
    
    # Save user message
    db.add_message(user_id, bot_id, "user", request.message)
    
    # Get conversation history
    history = db.conversations.get(user_id, {}).get(bot_id, [])
    chat_messages = [
        {"role": "system", "content": f"You are {bot.get('name')}. {bot.get('description')}"}
    ]
    
    # Add last 10 messages for context
    for msg in history[-10:]:
        chat_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    chat_messages.append({"role": "user", "content": request.message})
    
    # Get AI response
    ai_response = await openai_service.chat_completion(chat_messages, bot)
    
    # Save AI response
    db.add_message(user_id, bot_id, "assistant", ai_response)
    
    # Update bot message count
    db.bots[bot_id]["message_count"] += 1
    
    return {
        "success": True,
        "response": ai_response,
        "bot": bot["name"],
        "bot_id": bot_id,
        "message_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/chat/history/{bot_id}")
async def get_chat_history(bot_id: str, auth: dict = Depends(get_api_key)):
    """Get chat history with a specific bot"""
    user_id = auth["user_id"]
    
    if user_id not in db.conversations:
        return {"history": []}
    
    if bot_id not in db.conversations[user_id]:
        return {"history": []}
    
    return {
        "success": True,
        "bot_id": bot_id,
        "history": db.conversations[user_id][bot_id],
        "count": len(db.conversations[user_id][bot_id])
    }

@app.delete("/api/chat/history/{bot_id}")
async def clear_chat_history(bot_id: str, auth: dict = Depends(get_api_key)):
    """Clear chat history with a bot"""
    user_id = auth["user_id"]
    
    if user_id in db.conversations and bot_id in db.conversations[user_id]:
        db.conversations[user_id][bot_id] = []
    
    return {"success": True, "message": "Chat history cleared"}

# ==================== BOT MANAGEMENT ====================
@app.post("/api/bots")
async def create_bot(bot_data: BotCreate, auth: dict = Depends(get_api_key)):
    """Create a new AI bot"""
    bot_id = str(uuid.uuid4())
    
    bot = {
        "id": bot_id,
        "name": bot_data.name,
        "description": bot_data.description,
        "skills": bot_data.skills,
        "model": bot_data.model,
        "temperature": bot_data.temperature,
        "status": "online",
        "owner": auth["user_id"],
        "created_at": datetime.now().isoformat(),
        "message_count": 0,
        "is_public": bot_data.is_public,
        "avatar": bot_data.avatar
    }
    
    db.bots[bot_id] = bot
    
    return {
        "success": True,
        "bot": bot,
        "message": f"Bot '{bot_data.name}' created successfully",
        "chat_url": f"/chat-ui?bot={bot_id}"
    }

@app.get("/api/bots")
async def list_bots(auth: dict = Depends(get_api_key)):
    """List all available bots"""
    user_id = auth["user_id"]
    is_admin = auth.get("is_admin", False)
    
    # For admins, show all bots
    if is_admin:
        bots_list = list(db.bots.values())
    else:
        # For regular users, show public bots + their own bots
        bots_list = [
            bot for bot in db.bots.values()
            if bot["is_public"] or bot["owner"] == user_id
        ]
    
    return {
        "success": True,
        "count": len(bots_list),
        "bots": bots_list
    }

@app.get("/api/bots/{bot_id}")
async def get_bot(bot_id: str, auth: dict = Depends(get_api_key)):
    """Get specific bot details"""
    if bot_id not in db.bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = db.bots[bot_id]
    
    # Check permissions
    if not bot["is_public"] and bot["owner"] != auth["user_id"] and not auth["is_admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    return {
        "success": True,
        "bot": bot
    }

@app.put("/api/bots/{bot_id}")
async def update_bot(bot_id: str, bot_data: dict, auth: dict = Depends(get_api_key)):
    """Update bot configuration"""
    if bot_id not in db.bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = db.bots[bot_id]
    
    # Check ownership
    if bot["owner"] != auth["user_id"] and not auth["is_admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Update allowed fields
    allowed_fields = ["name", "description", "skills", "model", "temperature", "status", "avatar", "is_public"]
    for field in allowed_fields:
        if field in bot_data:
            bot[field] = bot_data[field]
    
    bot["updated_at"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "bot": bot,
        "message": f"Bot '{bot['name']}' updated"
    }

@app.delete("/api/bots/{bot_id}")
async def delete_bot(
    bot_id: str,
    override_token: Optional[str] = Body(None, embed=True),
    auth: dict = Depends(get_api_key)
):
    """Delete a bot (requires override token for non-admins)"""
    if bot_id not in db.bots:
        raise HTTPException(status_code=404, detail="Bot not found")
    
    bot = db.bots[bot_id]
    
    # Check ownership
    if bot["owner"] != auth["user_id"] and not auth["is_admin"]:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Non-admins need override token
    if not auth["is_admin"]:
        if not override_token or override_token != OVERRIDE_TOKEN:
            raise HTTPException(
                status_code=403,
                detail="Override token required for non-admin users"
            )
    
    # Delete bot
    del db.bots[bot_id]
    
    # Remove from conversations
    for user_id in db.conversations:
        if bot_id in db.conversations[user_id]:
            del db.conversations[user_id][bot_id]
    
    return {
        "success": True,
        "message": f"Bot '{bot['name']}' deleted",
        "bot_id": bot_id
    }

# ==================== CHAT UI ====================
@app.get("/chat-ui")
async def chat_interface():
    """Main chat interface (like DeepSeek)"""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Commander AI - Chat Interface</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            height: 100vh;
            overflow: hidden;
        }}
        
        .app-container {{
            display: flex;
            height: 100vh;
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        
        /* SIDEBAR */
        .sidebar {{
            width: 300px;
            background: #1a1a2e;
            color: white;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #2d2d4d;
        }}
        
        .sidebar-header {{
            padding: 25px;
            border-bottom: 1px solid #2d2d4d;
        }}
        
        .sidebar-header h1 {{
            font-size: 1.8rem;
            margin-bottom: 5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .sidebar-header p {{
            color: #a0a0c0;
            font-size: 0.9rem;
        }}
        
        .bot-list {{
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }}
        
        .bot-card {{
            background: #2d2d4d;
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }}
        
        .bot-card:hover {{
            background: #3d3d5d;
            transform: translateX(5px);
        }}
        
        .bot-card.active {{
            background: #3d3d5d;
            border-color: #667eea;
        }}
        
        .bot-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }}
        
        .bot-avatar {{
            width: 40px;
            height: 40px;
            background: #667eea;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
        }}
        
        .bot-info h3 {{
            font-size: 1.1rem;
            margin-bottom: 3px;
        }}
        
        .bot-status {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 0.8rem;
            color: #a0a0c0;
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
        }}
        
        .status-dot.offline {{
            background: #ef4444;
        }}
        
        .bot-skills {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 8px;
        }}
        
        .skill-tag {{
            background: #4f46e5;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }}
        
        .sidebar-footer {{
            padding: 20px;
            border-top: 1px solid #2d2d4d;
        }}
        
        .create-bot-btn {{
            width: 100%;
            padding: 12px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: transform 0.2s;
        }}
        
        .create-bot-btn:hover {{
            transform: translateY(-2px);
        }}
        
        /* MAIN CHAT AREA */
        .chat-area {{
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #f8fafc;
        }}
        
        .chat-header {{
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .chat-header-info {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .chat-header-avatar {{
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
        }}
        
        .chat-header-text h2 {{
            font-size: 1.5rem;
            margin-bottom: 3px;
        }}
        
        .chat-header-text p {{
            color: #64748b;
            font-size: 0.9rem;
        }}
        
        .chat-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .action-btn {{
            background: #f1f5f9;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            color: #64748b;
            transition: all 0.2s;
        }}
        
        .action-btn:hover {{
            background: #e2e8f0;
            color: #4f46e5;
        }}
        
        /* MESSAGES AREA */
        .messages-container {{
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .message {{
            max-width: 70%;
            padding: 15px 20px;
            border-radius: 18px;
            position: relative;
            animation: fadeIn 0.3s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .message.user {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 5px;
        }}
        
        .message.bot {{
            background: white;
            color: #333;
            align-self: flex-start;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .message-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        
        .message-avatar {{
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }}
        
        .message.user .message-avatar {{
            background: rgba(255,255,255,0.2);
        }}
        
        .message.bot .message-avatar {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        .message-sender {{
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .message-time {{
            font-size: 0.75rem;
            opacity: 0.7;
            margin-left: auto;
        }}
        
        .message-content {{
            line-height: 1.5;
        }}
        
        .message.bot .message-content {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .typing-indicator {{
            align-self: flex-start;
            background: white;
            padding: 15px 20px;
            border-radius: 18px;
            border-bottom-left-radius: 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .typing-dots {{
            display: flex;
            gap: 4px;
        }}
        
        .typing-dot {{
            width: 8px;
            height: 8px;
            background: #a0a0c0;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }}
        
        .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
        .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
        
        @keyframes typing {{
            0%, 60%, 100% {{ transform: translateY(0); }}
            30% {{ transform: translateY(-10px); }}
        }}
        
        /* INPUT AREA */
        .input-area {{
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e2e8f0;
        }}
        
        .input-container {{
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }}
        
        .message-input {{
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 25px;
            font-size: 1rem;
            font-family: inherit;
            resize: none;
            max-height: 120px;
            transition: border-color 0.2s;
        }}
        
        .message-input:focus {{
            outline: none;
            border-color: #4f46e5;
        }}
        
        .send-button {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.2s;
            font-size: 1.2rem;
        }}
        
        .send-button:hover:not(:disabled) {{
            transform: translateY(-2px);
        }}
        
        .send-button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        .input-actions {{
            display: flex;
            gap: 10px;
            margin-top: 10px;
            padding-left: 10px;
        }}
        
        .input-action {{
            background: none;
            border: none;
            color: #64748b;
            cursor: pointer;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.9rem;
            transition: all 0.2s;
        }}
        
        .input-action:hover {{
            background: #f1f5f9;
            color: #4f46e5;
        }}
        
        /* MODAL */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
        }}
        
        .modal-overlay.active {{
            opacity: 1;
            pointer-events: all;
        }}
        
        .modal {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            width: 500px;
            max-width: 90vw;
            max-height: 90vh;
            overflow-y: auto;
            transform: translateY(20px);
            transition: transform 0.3s;
        }}
        
        .modal-overlay.active .modal {{
            transform: translateY(0);
        }}
        
        .modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }}
        
        .modal-title {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .modal-close {{
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #64748b;
        }}
        
        .form-group {{
            margin-bottom: 20px;
        }}
        
        .form-label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #374151;
        }}
        
        .form-input {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 1rem;
            font-family: inherit;
            transition: border-color 0.2s;
        }}
        
        .form-input:focus {{
            outline: none;
            border-color: #4f46e5;
        }}
        
        .form-textarea {{
            min-height: 100px;
            resize: vertical;
        }}
        
        .form-select {{
            appearance: none;
            background: white url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%2364748b' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14 2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E") no-repeat right 15px center;
            padding-right: 45px;
        }}
        
        .modal-footer {{
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 30px;
        }}
        
        .btn {{
            padding: 12px 24px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4);
        }}
        
        .btn-secondary {{
            background: #f1f5f9;
            color: #64748b;
        }}
        
        .btn-secondary:hover {{
            background: #e2e8f0;
        }}
        
        /* RESPONSIVE */
        @media (max-width: 1024px) {{
            .app-container {{
                border-radius: 0;
                height: 100vh;
            }}
            
            .sidebar {{
                width: 250px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .sidebar {{
                display: none;
            }}
            
            .sidebar.mobile-visible {{
                display: flex;
                position: fixed;
                top: 0;
                left: 0;
                bottom: 0;
                z-index: 100;
                width: 100%;
            }}
            
            .mobile-menu-btn {{
                display: block;
            }}
        }}
        
        .mobile-menu-btn {{
            display: none;
            background: none;
            border: none;
            color: #64748b;
            font-size: 1.5rem;
            cursor: pointer;
        }}
        
        /* WELCOME SCREEN */
        .welcome-screen {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            text-align: center;
        }}
        
        .welcome-icon {{
            font-size: 4rem;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .welcome-title {{
            font-size: 2rem;
            margin-bottom: 10px;
            color: #1e293b;
        }}
        
        .welcome-text {{
            color: #64748b;
            max-width: 500px;
            margin-bottom: 30px;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="app-container">
        <!-- SIDEBAR -->
        <div class="sidebar" id="sidebar">
            <div class="sidebar-header">
                <h1>🤖 Commander AI</h1>
                <p>Chat with AI bots • Create your own • Deploy anywhere</p>
            </div>
            
            <div class="bot-list" id="botList">
                <!-- Bots will be loaded here -->
            </div>
            
            <div class="sidebar-footer">
                <button class="create-bot-btn" onclick="showCreateBotModal()">
                    <i class="fas fa-plus"></i> Create New Bot
                </button>
            </div>
        </div>
        
        <!-- MAIN CHAT AREA -->
        <div class="chat-area">
            <div class="chat-header">
                <button class="mobile-menu-btn" onclick="toggleSidebar()">
                    <i class="fas fa-bars"></i>
                </button>
                
                <div class="chat-header-info" id="chatHeader">
                    <div class="chat-header-avatar">🤖</div>
                    <div class="chat-header-text">
                        <h2>Welcome to Commander AI</h2>
                        <p>Select a bot to start chatting</p>
                    </div>
                </div>
                
                <div class="chat-actions">
                    <button class="action-btn" onclick="clearChat()" title="Clear chat">
                        <i class="fas fa-trash"></i>
                    </button>
                    <button class="action-btn" onclick="exportChat()" title="Export chat">
                        <i class="fas fa-download"></i>
                    </button>
                    <button class="action-btn" onclick="toggleDarkMode()" title="Toggle theme">
                        <i class="fas fa-moon"></i>
                    </button>
                </div>
            </div>
            
            <!-- MESSAGES CONTAINER -->
            <div class="messages-container" id="messagesContainer">
                <div class="welcome-screen" id="welcomeScreen">
                    <div class="welcome-icon">🤖</div>
                    <h2 class="welcome-title">Welcome to Commander AI</h2>
                    <p class="welcome-text">
                        Select a bot from the sidebar to start chatting.<br>
                        You can chat with AI assistants, create your own bots, or deploy them to the cloud.
                    </p>
                    <div style="display: flex; gap: 15px; margin-top: 20px;">
                        <button class="btn btn-primary" onclick="showCreateBotModal()">
                            <i class="fas fa-plus"></i> Create Your First Bot
                        </button>
                        <button class="btn btn-secondary" onclick="loadSampleBots()">
                            <i class="fas fa-robot"></i> Try Sample Bots
                        </button>
                    </div>
                </div>
                
                <!-- Messages will be loaded here -->
            </div>
            
            <!-- INPUT AREA -->
            <div class="input-area" id="inputArea" style="display: none;">
                <div class="input-container">
                    <textarea 
                        class="message-input" 
                        id="messageInput" 
                        placeholder="Type your message here..." 
                        rows="1"
                        onkeydown="handleInputKeydown(event)"
                    ></textarea>
                    <button class="send-button" id="sendButton" onclick="sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div class="input-actions">
                    <button class="input-action" onclick="insertCode()">
                        <i class="fas fa-code"></i> Code
                    </button>
                    <button class="input-action" onclick="uploadFile()">
                        <i class="fas fa-paperclip"></i> Attach
                    </button>
                    <button class="input-action" onclick="voiceInput()">
                        <i class="fas fa-microphone"></i> Voice
                    </button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- CREATE BOT MODAL -->
    <div class="modal-overlay" id="createBotModal">
        <div class="modal">
            <div class="modal-header">
                <h3 class="modal-title">🤖 Create New Bot</h3>
                <button class="modal-close" onclick="hideCreateBotModal()">&times;</button>
            </div>
            
            <form id="createBotForm" onsubmit="createNewBot(event)">
                <div class="form-group">
                    <label class="form-label">Bot Name</label>
                    <input type="text" class="form-input" id="botNameInput" placeholder="e.g., Code Assistant" required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Description</label>
                    <textarea class="form-input form-textarea" id="botDescriptionInput" placeholder="What does your bot do?" required></textarea>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Skills (comma separated)</label>
                    <input type="text" class="form-input" id="botSkillsInput" placeholder="e.g., coding, analysis, chat" required>
                </div>
                
                <div class="form-group">
                    <label class="form-label">AI Model</label>
                    <select class="form-input form-select" id="botModelInput">
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Fast & Cost-effective)</option>
                        <option value="gpt-4">GPT-4 (More powerful, slower)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Temperature (Creativity)</label>
                    <input type="range" class="form-input" id="botTemperatureInput" min="0" max="1" step="0.1" value="0.7">
                    <div style="display: flex; justify-content: space-between; margin-top: 5px;">
                        <span>Precise (0.0)</span>
                        <span id="temperatureValue">Creative (0.7)</span>
                        <span>Random (1.0)</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Avatar</label>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="🤖" checked> 🤖
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="💻"> 💻
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="📊"> 📊
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="🎨"> 🎨
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="🔬"> 🔬
                        </label>
                        <label style="cursor: pointer;">
                            <input type="radio" name="avatar" value="🚀"> 🚀
                        </label>
                    </div>
                </div>
                
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" onclick="hideCreateBotModal()">Cancel</button>
                    <button type="submit" class="btn btn-primary">Create Bot</button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- SCRIPT -->
    <script>
        // Configuration
        const API_KEY = "{CREATOR_API_KEY}";
        const OVERRIDE_TOKEN = "{OVERRIDE_TOKEN}";
        
        // State
        let currentBotId = null;
        let currentBot = null;
        let websocket = null;
        let userId = 'user-' + Math.random().toString(36).substr(2, 9);
        
        // DOM Elements
        const sidebar = document.getElementById('sidebar');
        const botList = document.getElementById('botList');
        const chatHeader = document.getElementById('chatHeader');
        const messagesContainer = document.getElementById('messagesContainer');
        const welcomeScreen = document.getElementById('welcomeScreen');
        const inputArea = document.getElementById('inputArea');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const createBotModal = document.getElementById('createBotModal');
        const temperatureInput = document.getElementById('botTemperatureInput');
        const temperatureValue = document.getElementById('temperatureValue');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            loadBots();
            setupEventListeners();
            
            // Auto-resize textarea
            messageInput.addEventListener('input', autoResizeTextarea);
        });
        
        // Event Listeners
        function setupEventListeners() {
            temperatureInput.addEventListener('input', (e) => {
                temperatureValue.textContent = `Creative (${{e.target.value}})`;
            });
        }
        
        // Bot Management
        async function loadBots() {
            try {
                const response = await fetch('/api/bots', {{
                    headers: {{
                        'X-API-Key': API_KEY
                    }}
                }});
                
                const data = await response.json();
                
                if (data.success) {
                    renderBotList(data.bots);
                }
            } catch (error) {
                console.error('Error loading bots:', error);
                showError('Failed to load bots');
            }
        }
        
        function renderBotList(bots) {
            botList.innerHTML = '';
            
            bots.forEach(bot => {{
                const botElement = document.createElement('div');
                botElement.className = 'bot-card';
                botElement.innerHTML = `
                    <div class="bot-header">
                        <div class="bot-avatar">${{bot.avatar}}</div>
                        <div class="bot-info">
                            <h3>${{bot.name}}</h3>
                            <div class="bot-status">
                                <span class="status-dot ${{bot.status === 'online' ? '' : 'offline'}}"></span>
                                <span>${{bot.status}}</span>
                            </div>
                        </div>
                    </div>
                    <p style="color: #a0a0c0; font-size: 0.9rem; margin-bottom: 10px;">${{bot.description}}</p>
                    <div class="bot-skills">
                        ${{bot.skills.map(skill => `<span class="skill-tag">${{skill}}</span>`).join('')}}
                    </div>
                `;
                
                botElement.addEventListener('click', () => selectBot(bot));
                botList.appendChild(botElement);
            }});
        }
        
        function selectBot(bot) {
            currentBotId = bot.id;
            currentBot = bot;
            
            // Update UI
            document.querySelectorAll('.bot-card').forEach(card => {{
                card.classList.remove('active');
            }});
            event.currentTarget.classList.add('active');
            
            // Update chat header
            chatHeader.innerHTML = `
                <div class="chat-header-avatar">${{bot.avatar}}</div>
                <div class="chat-header-text">
                    <h2>${{bot.name}}</h2>
                    <p>${{bot.description}}</p>
                </div>
            `;
            
            // Hide welcome screen
            welcomeScreen.style.display = 'none';
            inputArea.style.display = 'block';
            
            // Clear messages
            clearMessages();
            
            // Connect WebSocket
            connectWebSocket();
            
            // Load chat history
            loadChatHistory();
        }
        
        // WebSocket
        function connectWebSocket() {
            if (websocket) {{
                websocket.close();
            }}
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${{protocol}}//${{window.location.host}}/ws/chat/${{userId}}/${{currentBotId}}`;
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = () => {{
                console.log('WebSocket connected');
            }};
            
            websocket.onmessage = (event) => {{
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            }};
            
            websocket.onclose = () => {{
                console.log('WebSocket disconnected');
            }};
            
            websocket.onerror = (error) => {{
                console.error('WebSocket error:', error);
                showError('Connection error. Trying to reconnect...');
                setTimeout(connectWebSocket, 3000);
            }};
        }
        
        function handleWebSocketMessage(data) {
            switch (data.type) {
                case 'system':
                    addSystemMessage(data.content);
                    break;
                case 'message':
                    addBotMessage(data.content, data.bot_name, data.avatar);
                    break;
                case 'typing':
                    showTypingIndicator();
                    break;
            }
        }
        
        // Chat Functions
        function addSystemMessage(content) {
            const messageElement = document.createElement('div');
            messageElement.className = 'message system';
            messageElement.innerHTML = `
                <div class="message-content" style="text-align: center; color: #666;">
                    ${{content}}
                </div>
            `;
            messagesContainer.appendChild(messageElement);
            scrollToBottom();
        }
        
        function addUserMessage(content) {
            const messageElement = document.createElement('div');
            messageElement.className = 'message user';
            messageElement.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar">👤</div>
                    <div class="message-sender">You</div>
                    <div class="message-time">${{formatTime(new Date())}}</div>
                </div>
                <div class="message-content">${{escapeHtml(content)}}</div>
            `;
            messagesContainer.appendChild(messageElement);
            scrollToBottom();
        }
        
        function addBotMessage(content, botName, avatar) {
            hideTypingIndicator();
            
            const messageElement = document.createElement('div');
            messageElement.className = 'message bot';
            messageElement.innerHTML = `
                <div class="message-header">
                    <div class="message-avatar">${{avatar}}</div>
                    <div class="message-sender">${{botName}}</div>
                    <div class="message-time">${{formatTime(new Date())}}</div>
                </div>
                <div class="message-content">${{formatBotMessage(content)}}</div>
            `;
            messagesContainer.appendChild(messageElement);
            scrollToBottom();
        }
        
        function showTypingIndicator() {
            hideTypingIndicator();
            
            const typingElement = document.createElement('div');
            typingElement.className = 'typing-indicator';
            typingElement.id = 'typingIndicator';
            typingElement.innerHTML = `
                <div class="message-avatar">${{currentBot?.avatar || '🤖'}}</div>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            `;
            messagesContainer.appendChild(typingElement);
            scrollToBottom();
        }
        
        function hideTypingIndicator() {
            const existing = document.getElementById('typingIndicator');
            if (existing) {
                existing.remove();
            }
        }
        
        function sendMessage() {
            const content = messageInput.value.trim();
            if (!content || !websocket || websocket.readyState !== WebSocket.OPEN) {
                return;
            }
            
            // Add user message
            addUserMessage(content);
            
            // Send via WebSocket
            websocket.send(JSON.stringify({{
                type: 'message',
                content: content
            }}));
            
            // Clear input
            messageInput.value = '';
            autoResizeTextarea();
            
            // Disable send button temporarily
            sendButton.disabled = true;
            setTimeout(() => {{
                sendButton.disabled = false;
            }}, 1000);
        }
        
        async function loadChatHistory() {
            try {
                const response = await fetch(`/api/chat/history/${{currentBotId}}`, {{
                    headers: {{
                        'X-API-Key': API_KEY
                    }}
                }});
                
                const data = await response.json();
                
                if (data.success && data.history.length > 0) {
                    clearMessages();
                    
                    data.history.forEach(msg => {{
                        if (msg.role === 'user') {
                            addUserMessage(msg.content);
                        }} else if (msg.role === 'assistant') {{
                            addBotMessage(msg.content, currentBot.name, currentBot.avatar);
                        }}
                    }});
                }
            } catch (error) {
                console.error('Error loading chat history:', error);
            }
        }
        
        async function clearChat() {
            if (!currentBotId) return;
            
            if (confirm('Clear chat history with this bot?')) {
                try {
                    await fetch(`/api/chat/history/${{currentBotId}}`, {{
                        method: 'DELETE',
                        headers: {{
                            'X-API-Key': API_KEY
                        }}
                    }});
                    
                    clearMessages();
                    addSystemMessage('Chat history cleared');
                } catch (error) {
                    console.error('Error clearing chat:', error);
                    showError('Failed to clear chat');
                }
            }
        }
        
        // UI Functions
        function clearMessages() {
            const messages = messagesContainer.querySelectorAll('.message, .typing-indicator');
            messages.forEach(msg => msg.remove());
        }
        
        function autoResizeTextarea() {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
        }
        
        function handleInputKeydown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function formatTime(date) {
            return date.toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit' }});
        }
        
        function formatBotMessage(content) {
            // Convert markdown-like formatting
            let formatted = escapeHtml(content);
            
            // Code blocks
            formatted = formatted.replace(/```(\\w+)?\\n([\\s\\S]*?)\\n```/g, '<pre><code>$2</code></pre>');
            
            // Inline code
            formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Bold
            formatted = formatted.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');
            
            // Links
            formatted = formatted.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
            
            // Line breaks
            formatted = formatted.replace(/\\n/g, '<br>');
            
            return formatted;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Bot Creation
        function showCreateBotModal() {
            createBotModal.classList.add('active');
        }
        
        function hideCreateBotModal() {
            createBotModal.classList.remove('active');
        }
        
        async function createNewBot(event) {
            event.preventDefault();
            
            const botData = {{
                name: document.getElementById('botNameInput').value,
                description: document.getElementById('botDescriptionInput').value,
                skills: document.getElementById('botSkillsInput').value.split(',').map(s => s.trim()),
                model: document.getElementById('botModelInput').value,
                temperature: parseFloat(temperatureInput.value),
                avatar: document.querySelector('input[name="avatar"]:checked').value,
                is_public: true
            }};
            
            try {
                const response = await fetch('/api/bots', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                        'X-API-Key': API_KEY
                    }},
                    body: JSON.stringify(botData)
                }});
                
                const data = await response.json();
                
                if (data.success) {
                    hideCreateBotModal();
                    loadBots();
                    showSuccess('Bot created successfully!');
                    
                    // Reset form
                    event.target.reset();
                    temperatureValue.textContent = 'Creative (0.7)';
                    temperatureInput.value = 0.7;
                }} else {{
                    showError(data.detail || 'Failed to create bot');
                }}
            } catch (error) {
                console.error('Error creating bot:', error);
                showError('Failed to create bot');
            }
        }
        
        // Utility Functions
        function toggleSidebar() {
            sidebar.classList.toggle('mobile-visible');
        }
        
        function exportChat() {
            if (!currentBot) {{
                showError('Select a bot first');
                return;
            }}
            
            const messages = Array.from(messagesContainer.querySelectorAll('.message')).map(msg => {{
                const sender = msg.classList.contains('user') ? 'You' : currentBot.name;
                const content = msg.querySelector('.message-content').textContent;
                return `${{sender}}: ${{content}}`;
            }}).join('\\n\\n');
            
            const blob = new Blob([messages], {{ type: 'text/plain' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat-with-${{currentBot.name}}-${{new Date().toISOString().split('T')[0]}}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
        
        function toggleDarkMode() {
            document.body.classList.toggle('dark-mode');
        }
        
        function loadSampleBots() {
            showSuccess('Loading sample bots...');
            loadBots();
        }
        
        function insertCode() {
            const cursorPos = messageInput.selectionStart;
            const text = messageInput.value;
            const before = text.substring(0, cursorPos);
            const after = text.substring(cursorPos);
            messageInput.value = before + '```python\\n# Your code here\\n```' + after;
            autoResizeTextarea();
            messageInput.focus();
        }
        
        function uploadFile() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.txt,.pdf,.jpg,.png,.csv';
            input.onchange = (e) => {{
                const file = e.target.files[0];
                if (file) {{
                    showSuccess(`File selected: ${{file.name}}`);
                }}
            }};
            input.click();
        }
        
        function voiceInput() {
            if ('webkitSpeechRecognition' in window) {
                const recognition = new webkitSpeechRecognition();
                recognition.continuous = false;
                recognition.interimResults = false;
                
                recognition.onstart = () => {{
                    showSuccess('Listening...');
                }};
                
                recognition.onresult = (event) => {{
                    const transcript = event.results[0][0].transcript;
                    messageInput.value += transcript;
                    autoResizeTextarea();
                }};
                
                recognition.onerror = (event) => {{
                    showError('Speech recognition error');
                }};
                
                recognition.start();
            }} else {{
                showError('Speech recognition not supported');
            }}
        }
        
        // Notification Functions
        function showSuccess(message) {
            showNotification(message, 'success');
        }
        
        function showError(message) {
            showNotification(message, 'error');
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 10px;
                color: white;
                font-weight: 600;
                z-index: 1000;
                animation: slideIn 0.3s ease;
                ${{type === 'success' ? 'background: #10b981;' : 'background: #ef4444;'}}
            `;
            notification.textContent = message;
            document.body.appendChild(notification);
            
            setTimeout(() => {{
                notification.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }
        
        // Add CSS animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {{
                from {{ transform: translateX(100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            @keyframes slideOut {{
                from {{ transform: translateX(0); opacity: 1; }}
                to {{ transform: translateX(100%); opacity: 0; }}
            }}
            .dark-mode {{
                background: #1a1a2e;
            }}
            .dark-mode .app-container {{
                background: #2d2d4d;
                color: white;
            }}
            .dark-mode .chat-area {{
                background: #2d2d4d;
            }}
            .dark-mode .message.bot {{
                background: #3d3d5d;
                color: white;
            }}
            .dark-mode .message-input {{
                background: #3d3d5d;
                color: white;
                border-color: #4f46e5;
            }}
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
    """
    return HTMLResponse(html)

# ==================== KEEP-ALIVE FOR RENDER ====================
import threading
import requests

def keep_alive():
    """Ping the app every 10 minutes to prevent Render sleep"""
    while True:
        try:
            # Try to get Render external URL
            external_url = os.environ.get("RENDER_EXTERNAL_URL", "")
            if external_url:
                requests.get(f"{external_url}/health", timeout=10)
                print(f"✅ Keep-alive ping sent to {external_url}")
            else:
                # Try local health check
                requests.get(f"http://localhost:{PORT}/health", timeout=5)
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")
        
        # Sleep for 10 minutes (Render sleeps after 15 minutes)
        time.sleep(600)

# Start keep-alive in background
threading.Thread(target=keep_alive, daemon=True).start()

# ==================== START SERVER ====================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 COMMANDER AI CHAT - STARTING")
    print("=" * 60)
    print(f"📡 Port: {PORT}")
    print(f"👑 Creator: {CREATOR_EMAIL}")
    print(f"🔑 API Key: {CREATOR_API_KEY}")
    print(f"🔐 Override Token: {OVERRIDE_TOKEN}")
    print(f"🤖 OpenAI: {'✅ ENABLED' if openai_service.enabled else '❌ DISABLED'}")
    print(f"💬 Chat UI: http://localhost:{PORT}/chat-ui")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"❤️  Health: http://localhost:{PORT}/health")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",  # Must be 0.0.0.0 for Render
        port=PORT,
        log_level="info"
    )
