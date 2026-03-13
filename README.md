# nexus.ai
An opensource free to use for non-expert in coding and putting puzzle together.
#!/usr/bin/env python3
"""
NexusAI - Terminal AI Platform
==============================
A unified AI platform combining:
- Local models (Ollama, LM Studio, llama.cpp)
- Cloud models (HuggingFace, GitHub, Ollama.com)
- Chat + Coding + Automation + RAG

Usage:
    python3 nexusai.py              # Start interactive mode
    python3 nexusai.py --chat       # Quick chat
    python3 nexusai.py --code       # Coding mode
    python3 nexusai.py --rag        # Document Q&A
    python3 nexusai.py --list      # List available models
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

# Core imports
from core.model_router import ModelRouter
from core.chat import ChatSession
from tools.coder import CoderTools
from tools.rag import RAGEngine

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


VERSION = "1.0.0"
BANNER = f"""
{Fore.CYAN}╔═══════════════════════════════════════════╗
║        {Fore.YELLOW}NexusAI{Fore.CYAN} v{VERSION}                      ║
║   {Fore.WHITE}Unified Local + Cloud AI Platform{Fore.CYAN}        ║
╚═══════════════════════════════════════════╝{Style.RESET_ALL}

{Fore.GREEN}Local:{Fore.WHITE} Ollama, LM Studio, llama.cpp, GPT4All
{Fore.BLUE}Cloud:{Fore.WHITE} HuggingFace, GitHub, Ollama.com
{Fore.MAGENTA}Features:{Fore.WHITE} Chat • Code • RAG • Agents
"""


class NexusAI:
    """Main NexusAI application."""
    
    def __init__(self):
        self.router = ModelRouter()
        self.chat = ChatSession(self.router)
        self.coder = CoderTools(self.router)
        self.rag = RAGEngine()
        self.running = True
        
    def print_status(self):
        """Print current status."""
        local = self.router.get_local_models()
        cloud = self.router.get_cloud_models()
        
        print(f"\n{Fore.CYAN}=== Status ==={Fore.RESET}")
        print(f"Local models:  {len(local)}")
        print(f"Cloud models:  {len(cloud)}")
        print(f"Default:       {self.router.default_model}")
        
    def list_models(self):
        """List all available models."""
        print(f"\n{Fore.CYAN}=== Available Models ==={Fore.RESET}\n")
        
        print(f"{Fore.GREEN}Local Models:{Fore.RESET}")
        for m in self.router.get_local_models():
            print(f"  • {m}")
            
        print(f"\n{Fore.BLUE}Cloud Models:{Fore.RESET}")
        for m in self.router.get_cloud_models():
            print(f"  • {m}")
            
    def chat_mode(self, prompt: str = None):
        """Interactive chat mode."""
        print(f"\n{Fore.YELLOW}Chat Mode - Type 'exit' to quit{Fore.RESET}\n")
        
        while self.running:
            if prompt:
                user_input = prompt
                prompt = None
            else:
                user_input = input(f"{Fore.GREEN}➜ {Fore.RESET}").strip()
                
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit', 'q']:
                break
            if user_input.lower() == '/status':
                self.print_status()
                continue
            if user_input.lower() == '/models':
                self.list_models()
                continue
                
            response = self.chat.send(user_input)
            print(f"\n{Fore.CYAN}NexusAI:{Fore.RESET} {response}\n")
            
    def code_mode(self, task: str = None):
        """Coding assistance mode."""
        print(f"\n{Fore.YELLOW}Code Mode - Describe what you want to build{Fore.RESET}\n")
        
        if task:
            result = self.coder.execute_task(task)
            print(f"\n{Fore.CYAN}Result:{Fore.RESET}\n{result}")
        
    def rag_mode(self, query: str = None):
        """Document Q&A mode."""
        print(f"\n{Fore.YELLOW}RAG Mode - Ask questions about your documents{Fore.RESET}\n")
        
        if query:
            result = self.rag.query(query)
            print(f"\n{Fore.CYAN}Answer:{Fore.RESET}\n{result}")
            
    def run(self, args):
        """Main run loop."""
        print(BANNER)
        
        if args.list:
            self.list_models()
            return
            
        if args.status:
            self.print_status()
            return
            
        if args.chat:
            self.chat_mode(args.chat)
            return
            
        if args.code:
            self.code_mode(args.code)
            return
            
        if args.rag:
            self.rag_mode(args.rag)
            return
            
        # Interactive mode
        self.chat_mode()


def main():
    parser = argparse.ArgumentParser(description="NexusAI - Unified AI Platform")
    parser.add_argument('--chat', '-c', nargs='?', const='', help='Chat mode')
    parser.add_argument('--code', nargs='?', const='', help='Code mode')
    parser.add_argument('--rag', '-r', nargs='?', const='', help='RAG mode')
    parser.add_argument('--list', '-l', action='store_true', help='List models')
    parser.add_argument('--status', '-s', action='store_true', help='Show status')
    parser.add_argument('--version', '-v', action='store_true', help='Version')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"NexusAI v{VERSION}")
        return
        
    app = NexusAI()
    app.run(args)


if __name__ == "__main__":
    main()
"""
NexusAI - Chat Session
======================
Handles chat conversations with context.
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


class ChatSession:
    """Manages chat sessions with history."""
    
    def __init__(self, router):
        self.router = router
        self.history: List[Message] = []
        self.system_prompt = "You are NexusAI, a helpful AI assistant."
        
    def send(self, prompt: str, system: str = None) -> str:
        """Send message and get response."""
        if system:
            self.system_prompt = system
            
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend([{"role": m.role, "content": m.content} for m in self.history[-10:]])
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.router.chat(prompt)
            self.history.append(Message("user", prompt))
            self.history.append(Message("assistant", response))
            return response
        except Exception as e:
            return f"Error: {str(e)}"
            
    def clear(self):
        """Clear chat history."""
        self.history = []
        
    def get_history(self) -> List[Dict]:
        """Get chat history."""
        return [{"role": m.role, "content": m.content} for m in self.history]
        
    def save_history(self, filename: str):
        """Save chat history to file."""
        import json
        with open(filename, "w") as f:
            json.dump(self.get_history(), f, indent=2)
            
    def load_history(self, filename: str):
        """Load chat history from file."""
        import json
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                self.history = [Message(m["role"], m["content"]) for m in data]
        except FileNotFoundError:
            pass
# NexusAI Core
"""
NexusAI - Coding Tools
======================
Code generation, debugging, and file operations.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class CoderTools:
    """Coding assistance tools."""
    
    def __init__(self, router):
        self.router = router
        
    def execute_task(self, task: str) -> str:
        """Execute a coding task."""
        # Analyze the task
        prompt = f"""You are a coding assistant. Complete this task:
{task}

Provide:
1. The code solution
2. Explanation of how it works
3. Any required dependencies"""

        return self.router.chat(prompt, model="qwen2.5-coder:14b")
        
    def read_file(self, filepath: str) -> str:
        """Read a file's contents."""
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {filepath}: {e}"
            
    def write_file(self, filepath: str, content: str) -> str:
        """Write content to a file."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(content)
            return f"Written to {filepath}"
        except Exception as e:
            return f"Error writing {filepath}: {e}"
            
    def run_command(self, cmd: str) -> str:
        """Run a shell command."""
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout or result.stderr
            return f"Output:\n{output}"
        except Exception as e:
            return f"Error: {e}"
            
    def explain_code(self, code: str) -> str:
        """Explain what code does."""
        prompt = f"""Explain this code in simple terms:
```{code}
```"""
        return self.router.chat(prompt)
        
    def debug_code(self, code: str, error: str = None) -> str:
        """Debug code and suggest fixes."""
        prompt = f"""Debug this code:
```{code}
```
Error: {error or 'None provided'}
Suggest fixes and explain the issue."""
        return self.router.chat(prompt)
        
    def generate_tests(self, code: str) -> str:
        """Generate unit tests for code."""
        prompt = f"""Generate unit tests for this code:
```{code}
```"""
        return self.router.chat(prompt)
