# llm_client.py
import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

class CodeDebugAgent:
    """AI-powered code debugging agent using MCP and ChatGroq"""
    
    def __init__(self, groq_api_key: str = GROQ_API_KEY, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the code debug agent.
        
        Args:
            groq_api_key: Your Groq API key
            model: Groq model to use for AI responses
        """
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model,
            temperature=0.1,  # Low temperature for more deterministic code analysis
            max_tokens=4000
        )
        
        self.session: Optional[ClientSession] = None
        self.conversation_history: List[Dict[str, str]] = []
        
        # MCP server parameters
        self.server_params = StdioServerParameters(
            command="python",
            args=[str(Path(__file__).parent / "code_debug_server.py")],
            env=None
        )
    
    async def start_mcp_connection(self):
        """Start connection to MCP server"""
        print("🔌 Connecting to Code Debug MCP Server...")
        
        try:
            # Create a new context for the stdio client
            self.stdio_context = stdio_client(self.server_params)
            self.read_stream, self.write_stream = await self.stdio_context.__aenter__()
            
            # Create a new context for the client session
            self.session_context = ClientSession(self.read_stream, self.write_stream)
            self.session = await self.session_context.__aenter__()
            
            # Initialize the session
            await self.session.initialize()
            
            print("✅ Connected to MCP server successfully!")
            
            # List available tools and resources
            tools = await self.session.list_tools()
            resources = await self.session.list_resources()
            prompts = await self.session.list_prompts()
            
            print(f"📚 Available tools: {len(tools.tools)}")
            print(f"📂 Available resources: {len(resources.resources) if resources.resources else 0}")
            print(f"💭 Available prompts: {len(prompts.prompts)}")
            
        except Exception as e:
            print(f"❌ Error: Connection closed")
            await self.close_connection()
            raise RuntimeError(f"Failed to connect to MCP server: {str(e)}")
    
    async def close_connection(self):
        """Close MCP connection"""
        try:
            if hasattr(self, 'session') and self.session:
                await self.session_context.__aexit__(None, None, None)
                self.session = None
            
            if hasattr(self, 'read_stream') and self.read_stream:
                await self.stdio_context.__aexit__(None, None, None)
                self.read_stream = None
                self.write_stream = None
                
            print("🔌 MCP connection closed")
        except Exception as e:
            print(f"Warning: Error during connection cleanup: {str(e)}")
    
    async def list_project_files(self, directory: str = ".", extensions: str = "py,js,ts") -> str:
        """List code files in the project"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.session.call_tool(
                "list_project_files",
                arguments={"directory": directory, "file_extensions": extensions}
            )
            return result.content[0].text if result.content else "No files found"
        except Exception as e:
            return f"Error listing files: {str(e)}"
    
    async def read_code_file(self, file_path: str) -> str:
        """Read code file content via MCP resource"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            content, mime_type = await self.session.read_resource(f"code://{file_path}")
            return content
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def review_code_with_ai(self, file_path: str, focus_areas: str = "general") -> str:
        """Perform AI-powered code review"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Get code review from MCP server
            review_result = await self.session.call_tool(
                "review_code",
                arguments={"file_path": file_path, "focus_areas": focus_areas}
            )
            
            review_content = review_result.content[0].text if review_result.content else ""
            
            # Get AI analysis
            prompt = f"""You are an expert code reviewer. Please analyze the following code review report and provide additional insights:

{review_content}

Please provide:
1. Summary of key findings
2. Priority ranking of issues (High/Medium/Low)
3. Specific actionable recommendations
4. Code examples for suggested improvements
5. Overall code quality assessment (1-10 scale)

Be constructive and specific in your feedback."""

            ai_response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            combined_review = f"""# AI-Enhanced Code Review

## MCP Server Analysis:
{review_content}

## AI Expert Analysis:
{ai_response.content}
"""
            
            return combined_review
            
        except Exception as e:
            return f"Error during code review: {str(e)}"
    
    async def debug_code_with_ai(self, file_path: str, issue_description: str) -> Dict[str, str]:
        """Debug code with AI assistance"""
        if not self.session:
            return {"error": "Not connected to MCP server"}
        
        try:
            # Get debug analysis from MCP server
            debug_result = await self.session.call_tool(
                "debug_and_fix_code",
                arguments={
                    "file_path": file_path,
                    "issue_description": issue_description,
                    "auto_analyze": True
                }
            )
            
            debug_content = debug_result.content[0].text if debug_result.content else ""
            
            # Get AI analysis
            prompt = f"""You are an expert software debugger. Please analyze this debugging report and provide additional insights:

{debug_content}

Please provide:
1. Root cause analysis
2. Step-by-step debugging process
3. Complete fixed code
4. Explanation of changes
5. Prevention strategies

Be thorough and provide working code that can be directly applied."""

            ai_response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            
            return {
                "analysis": ai_response.content,
                "server_analysis": debug_content,
                "file_path": file_path
            }
            
        except Exception as e:
            return {"error": f"Error during debugging: {str(e)}"}
    
    async def apply_code_fix(self, file_path: str, issue_description: str, fixed_code: str, backup: bool = True) -> str:
        """Apply the AI-suggested code fix"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.session.call_tool(
                "debug_and_fix_code",
                arguments={
                    "file_path": file_path,
                    "issue_description": issue_description,
                    "proposed_fix": fixed_code,
                    "backup": backup
                }
            )
            
            return result.content[0].text if result.content else "Fix applied successfully"
            
        except Exception as e:
            return f"Error applying fix: {str(e)}"
    
    async def modify_code_section(self, file_path: str, start_line: int, end_line: int, new_code: str, description: str = "Code modification", backup: bool = True) -> str:
        """Modify a specific section of code"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.session.call_tool(
                "modify_code_section",
                arguments={
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "new_code": new_code,
                    "description": description,
                    "backup": backup
                }
            )
            
            return result.content[0].text if result.content else "Code section modified successfully"
            
        except Exception as e:
            return f"Error modifying code section: {str(e)}"
    
    async def clear_cache(self) -> str:
        """Clear server caches"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.session.call_tool("clear_cache")
            return result.content[0].text if result.content else "Cache cleared successfully"
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    
    async def write_ai_code(self, file_path: str, code_content: str, language: str = "python", overwrite: bool = False, backup: bool = True) -> str:
        """Write AI-generated code to a file"""
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            result = await self.session.call_tool(
                "write_ai_code",
                arguments={
                    "file_path": file_path,
                    "code_content": code_content,
                    "language": language,
                    "overwrite": overwrite,
                    "backup": backup
                }
            )
            
            return result.content[0].text if result.content else "Code written successfully"
            
        except Exception as e:
            return f"Error writing code: {str(e)}"
    
    async def generate_and_write_code(self, instruction: str, file_path: str, language: str = "python") -> str:
        """
        Autonomously generate code based on instruction and write to file
        
        Args:
            instruction: High-level description of what code to generate
            file_path: Where to write the generated code
            language: Programming language
        """
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Create a comprehensive prompt for code generation
            generation_prompt = f"""You are an expert {language} developer. Generate complete, production-ready code based on this instruction:

INSTRUCTION: {instruction}

Requirements:
1. Generate complete, working {language} code
2. Include proper error handling
3. Add appropriate comments and docstrings
4. Follow {language} best practices and conventions
5. Make the code production-ready
6. Include imports and dependencies as needed

Please provide ONLY the code without explanations or markdown formatting. The code should be ready to write directly to a file."""

            # Get AI-generated code
            ai_response = await self.llm.ainvoke([HumanMessage(content=generation_prompt)])
            generated_code = ai_response.content.strip()
            
            # Clean up any markdown formatting that might have been added
            if generated_code.startswith(f"```{language}"):
                generated_code = generated_code[len(f"```{language}"):].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3].strip()
            
            # Write the generated code using MCP
            result = await self.write_ai_code(
                file_path=file_path,
                code_content=generated_code,
                language=language,
                overwrite=False,
                backup=True
            )
            
            return f"""Code Generated and Written Successfully!

**Instruction**: {instruction}
**File**: {file_path}
**Language**: {language}

**Generated Code**:
```{language}
{generated_code}
```

**Write Result**:
{result}
"""
            
        except Exception as e:
            return f"Error in autonomous code generation: {str(e)}"

    async def analyze_and_refactor_code(self, file_path: str, refactor_goals: str) -> str:
        """
        Autonomously analyze code and generate refactored version
        
        Args:
            file_path: Path to code file to refactor
            refactor_goals: What to improve (e.g., "improve performance", "add error handling")
        """
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # First, read the existing code
            current_code = await self.read_code_file(file_path)
            
            # Create refactoring prompt
            refactor_prompt = f"""You are an expert software engineer. Analyze and refactor the following code:

CURRENT CODE:
{current_code}

REFACTORING GOALS: {refactor_goals}

Please:
1. Analyze the current code structure and identify issues
2. Generate improved, refactored code that addresses the goals
3. Maintain the same functionality while improving the code
4. Add comments explaining major changes

Provide ONLY the complete refactored code without explanations or markdown formatting."""

            # Get AI refactored code
            ai_response = await self.llm.ainvoke([HumanMessage(content=refactor_prompt)])
            refactored_code = ai_response.content.strip()
            
            # Clean up formatting
            language = self.get_file_language(file_path)
            if refactored_code.startswith(f"```{language}"):
                refactored_code = refactored_code[len(f"```{language}"):].strip()
            if refactored_code.endswith("```"):
                refactored_code = refactored_code[:-3].strip()
            
            # Apply the refactored code
            result = await self.write_ai_code(
                file_path=file_path,
                code_content=refactored_code,
                language=language,
                overwrite=True,
                backup=True
            )
            
            return f"""Code Refactored Successfully!

**File**: {file_path}
**Goals**: {refactor_goals}

**Refactored Code**:
```{language}
{refactored_code}
```

**Write Result**:
{result}
"""
            
        except Exception as e:
            return f"Error in autonomous refactoring: {str(e)}"

    async def intelligent_bug_fix(self, file_path: str, auto_apply: bool = True) -> str:
        """
        Autonomously detect and fix bugs in code
        
        Args:
            file_path: Path to code file
            auto_apply: Whether to automatically apply the fix
        """
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Get debug analysis first
            debug_result = await self.session.call_tool(
                "debug_and_fix_code",
                arguments={
                    "file_path": file_path,
                    "auto_analyze": True
                }
            )
            
            debug_content = debug_result.content[0].text if debug_result.content else ""
            
            # Get the current code
            current_code = await self.read_code_file(file_path)
            
            # Create intelligent bug fixing prompt
            fix_prompt = f"""You are an expert debugger and software engineer. Analyze the following code and debug report, then generate a fixed version:

CURRENT CODE:
{current_code}

DEBUG ANALYSIS:
{debug_content}

Your task:
1. Identify all bugs and issues mentioned in the analysis
2. Generate completely fixed code that resolves all issues
3. Preserve the original functionality while fixing bugs
4. Add comments explaining the fixes made
5. Ensure the code follows best practices

Provide ONLY the complete fixed code without explanations or markdown formatting."""

            # Get AI-generated fix
            ai_response = await self.llm.ainvoke([HumanMessage(content=fix_prompt)])
            fixed_code = ai_response.content.strip()
            
            # Clean up formatting
            language = self.get_file_language(file_path)
            if fixed_code.startswith(f"```{language}"):
                fixed_code = fixed_code[len(f"```{language}"):].strip()
            if fixed_code.endswith("```"):
                fixed_code = fixed_code[:-3].strip()
            
            if auto_apply:
                # Apply the fix automatically
                result = await self.apply_code_fix(
                    file_path=file_path,
                    issue_description="Autonomous bug fix based on analysis",
                    fixed_code=fixed_code,
                    backup=True
                )
                
                return f"""Bugs Fixed Automatically!

**File**: {file_path}

**Fixed Code**:
```{language}
{fixed_code}
```

**Apply Result**:
{result}
"""
            else:
                return f"""Bug Fix Generated (Not Applied):

**File**: {file_path}

**Suggested Fix**:
```{language}
{fixed_code}
```

**Debug Analysis**:
{debug_content}

Call with auto_apply=True to apply the fix automatically.
"""
            
        except Exception as e:
            return f"Error in intelligent bug fixing: {str(e)}"

    async def autonomous_development_session(self, project_requirements: str) -> str:
        """
        Completely autonomous development session based on high-level requirements
        
        Args:
            project_requirements: High-level description of what to build
        """
        if not self.session:
            return "Error: Not connected to MCP server"
        
        try:
            # Plan the project structure
            planning_prompt = f"""You are a senior software architect. Based on these requirements, create a development plan:

REQUIREMENTS: {project_requirements}

Please provide:
1. List of files to create (with file paths)
2. Brief description of what each file should contain
3. Development order (which files to create first)
4. Key components and their responsibilities

Format your response as JSON:
{{
    "files": [
        {{"path": "main.py", "description": "Main application entry point", "priority": 1}},
        {{"path": "utils.py", "description": "Utility functions", "priority": 2}}
    ],
    "overview": "Brief project overview"
}}"""

            # Get development plan
            plan_response = await self.llm.ainvoke([HumanMessage(content=planning_prompt)])
            
            # Extract plan (simplified - you might want to use JSON parsing)
            plan_content = plan_response.content
            
            results = []
            results.append(f"🚀 Starting Autonomous Development Session")
            results.append(f"📋 Requirements: {project_requirements}")
            results.append(f"📝 Development Plan:\n{plan_content}")
            
            # For demonstration, let's create a simple example
            # In a real implementation, you'd parse the JSON and create files accordingly
            
            # Example: Create a main.py file based on requirements
            main_file_result = await self.generate_and_write_code(
                instruction=f"Create a main.py file that serves as the entry point for: {project_requirements}",
                file_path="main.py",
                language="python"
            )
            results.append(f"\n📄 Created main.py:\n{main_file_result}")
            
            # Example: Create a utils.py file
            utils_file_result = await self.generate_and_write_code(
                instruction=f"Create a utils.py file with helper functions for: {project_requirements}",
                file_path="utils.py", 
                language="python"
            )
            results.append(f"\n📄 Created utils.py:\n{utils_file_result}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"Error in autonomous development: {str(e)}"

    def get_file_language(self, file_path: str) -> str:
        """Helper method to determine file language"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby'
        }
        return language_map.get(ext, 'text')

    async def interactive_debug_session(self):
        """Start an interactive debugging session"""
        print("\n🤖 Welcome to the AI Code Debug Agent!")
        print("Type 'help' for available commands, 'quit' to exit\n")
        
        while True:
            try:
                command = input("🔧 Debug Agent > ").strip()
                
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'help':
                    self.show_help()
                elif command.startswith('list'):
                    # list [directory] [extensions]
                    parts = command.split()
                    directory = parts[1] if len(parts) > 1 else "."
                    extensions = parts[2] if len(parts) > 2 else "py,js,ts"
                    result = await self.list_project_files(directory, extensions)
                    print(result)
                elif command.startswith('review'):
                    # review <file_path> [focus_areas]
                    parts = command.split(maxsplit=2)
                    if len(parts) < 2:
                        print("Usage: review <file_path> [focus_areas]")
                        continue
                    file_path = parts[1]
                    focus_areas = parts[2] if len(parts) > 2 else "general"
                    
                    print(f"🔍 Reviewing {file_path}...")
                    result = await self.review_code_with_ai(file_path, focus_areas)
                    print(result)
                elif command.startswith('debug'):
                    # debug <file_path>
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: debug <file_path>")
                        continue
                    file_path = parts[1]
                    
                    issue_description = input("🐛 Describe the issue: ").strip()
                    if not issue_description:
                        print("Issue description is required!")
                        continue
                    
                    print(f"🔍 Debugging {file_path}...")
                    result = await self.debug_code_with_ai(file_path, issue_description)
                    
                    if "error" in result:
                        print(f"❌ {result['error']}")
                    else:
                        print(result["analysis"])
                        
                        # Ask if user wants to apply the fix
                        apply_fix = input("\n🔧 Apply the suggested fix? (y/n): ").strip().lower()
                        if apply_fix == 'y':
                            # Extract the fixed code from AI response (this is simplified)
                            print("📝 Please provide the corrected code:")
                            print("(End with '###END###' on a new line)")
                            
                            fixed_code_lines = []
                            while True:
                                line = input()
                                if line.strip() == "###END###":
                                    break
                                fixed_code_lines.append(line)
                            
                            fixed_code = "\n".join(fixed_code_lines)
                            
                            if fixed_code.strip():
                                fix_result = await self.apply_code_fix(file_path, issue_description, fixed_code)
                                print(fix_result)
                            else:
                                print("No code provided, fix not applied.")
                elif command.startswith('read'):
                    # read <file_path>
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: read <file_path>")
                        continue
                    file_path = parts[1]
                    
                    content = await self.read_code_file(file_path)
                    print(content)
                elif command.startswith('modify'):
                    # modify <file_path> <start_line> <end_line>
                    parts = command.split()
                    if len(parts) < 4:
                        print("Usage: modify <file_path> <start_line> <end_line>")
                        continue
                    
                    file_path = parts[1]
                    try:
                        start_line = int(parts[2])
                        end_line = int(parts[3])
                    except ValueError:
                        print("Error: Start and end lines must be numbers")
                        continue
                    
                    print("📝 Enter the new code (End with '###END###' on a new line):")
                    new_code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "###END###":
                            break
                        new_code_lines.append(line)
                    
                    new_code = "\n".join(new_code_lines)
                    if new_code.strip():
                        result = await self.modify_code_section(file_path, start_line, end_line, new_code)
                        print(result)
                    else:
                        print("No code provided, modification cancelled.")
                elif command.startswith('write'):
                    # write <file_path> [language]
                    parts = command.split(maxsplit=2)
                    if len(parts) < 2:
                        print("Usage: write <file_path> [language]")
                        continue
                    
                    file_path = parts[1]
                    language = parts[2] if len(parts) > 2 else "python"
                    
                    print("📝 Enter the code to write (End with '###END###' on a new line):")
                    code_lines = []
                    while True:
                        line = input()
                        if line.strip() == "###END###":
                            break
                        code_lines.append(line)
                    
                    code = "\n".join(code_lines)
                    if code.strip():
                        result = await self.write_ai_code(file_path, code, language)
                        print(result)
                    else:
                        print("No code provided, write cancelled.")
                elif command.startswith('generate'):
                    # generate <file_path> [language]
                    parts = command.split(maxsplit=2)
                    if len(parts) < 2:
                        print("Usage: generate <file_path> [language]")
                        continue
                    
                    file_path = parts[1]
                    language = parts[2] if len(parts) > 2 else "python"
                    
                    print("📝 Enter the code description (End with '###END###' on a new line):")
                    description_lines = []
                    while True:
                        line = input()
                        if line.strip() == "###END###":
                            break
                        description_lines.append(line)
                    
                    description = "\n".join(description_lines)
                    if description.strip():
                        result = await self.generate_and_write_code(description, file_path, language)
                        print(result)
                    else:
                        print("No description provided, generation cancelled.")
                elif command.startswith('refactor'):
                    # refactor <file_path>
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: refactor <file_path>")
                        continue
                    
                    file_path = parts[1]
                    print("📝 Enter refactoring goals (End with '###END###' on a new line):")
                    goals_lines = []
                    while True:
                        line = input()
                        if line.strip() == "###END###":
                            break
                        goals_lines.append(line)
                    
                    goals = "\n".join(goals_lines)
                    if goals.strip():
                        result = await self.analyze_and_refactor_code(file_path, goals)
                        print(result)
                    else:
                        print("No goals provided, refactoring cancelled.")
                elif command.startswith('fix'):
                    # fix <file_path> [auto]
                    parts = command.split()
                    if len(parts) < 2:
                        print("Usage: fix <file_path> [auto]")
                        continue
                    
                    file_path = parts[1]
                    auto_apply = len(parts) > 2 and parts[2].lower() == 'auto'
                    
                    print(f"🔍 Analyzing {file_path} for bugs...")
                    result = await self.intelligent_bug_fix(file_path, auto_apply)
                    print(result)
                elif command.startswith('dev'):
                    # dev <requirements>
                    parts = command.split(maxsplit=1)
                    if len(parts) < 2:
                        print("Usage: dev <requirements>")
                        continue
                    
                    requirements = parts[1]
                    print(f"🚀 Starting autonomous development for: {requirements}")
                    result = await self.autonomous_development_session(requirements)
                    print(result)
                elif command == 'clear-cache':
                    result = await self.clear_cache()
                    print(result)
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    def show_help(self):
        """Show available commands"""
        help_text = """
Available Commands:
==================

📁 list [directory] [extensions]  - List code files in directory
   Example: list . py,js
   
🔍 review <file_path> [focus_areas] - AI-powered code review
   Example: review main.py security,performance
   
🐛 debug <file_path>               - Debug code issues with AI
   Example: debug app.py
   
📖 read <file_path>                - Read and display code file
   Example: read utils.py
   
✏️ modify <file_path> <start> <end> - Modify specific code section
   Example: modify main.py 10 20
   
📝 write <file_path> [language]    - Write new code file
   Example: write new.py python

🤖 Autonomous Features:
======================
generate <file_path> <language>    - Generate code from description
   Example: generate app.py python
   (You'll be prompted for the code description)

refactor <file_path>              - Refactor code with AI
   Example: refactor main.py
   (You'll be prompted for refactoring goals)

fix <file_path> [auto]            - Automatically fix bugs
   Example: fix buggy.py auto
   (Add 'auto' to automatically apply fixes)

dev <requirements>                - Start autonomous development
   Example: dev "Create a REST API"
   (You'll be prompted for project requirements)

🗑️ clear-cache                    - Clear server caches
   
❓ help                           - Show this help message
🚪 quit/exit/q                    - Exit the debug agent

Focus Areas for Review:
======================
- general, bugs, security, performance, style, testing
"""
        print(help_text)


async def main():
    """Main function to run the code debug agent"""
    # Get Groq API key from environment or user input
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = input("Enter your Groq API key: ").strip()
        if not groq_api_key:
            print("Groq API key is required!")
            return
    
    # Initialize the agent
    agent = None
    try:
        agent = CodeDebugAgent(groq_api_key)
        
        # Start MCP connection
        await agent.start_mcp_connection()
        
        # Start interactive session
        await agent.interactive_debug_session()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if agent:
            try:
                await agent.close_connection()
            except Exception as e:
                print(f"Warning: Error during cleanup: {str(e)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {str(e)}")