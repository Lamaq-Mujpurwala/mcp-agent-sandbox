# code_debug_server.py
import os
import ast
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import functools
import hashlib
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP, Context
from dataclasses import dataclass


@dataclass
class ServerContext:
    """Server context for managing state"""
    project_root: Path
    allowed_extensions: set[str]
    file_cache: dict[str, tuple[str, float, str]] = None  # path -> (content, timestamp, hash)
    analysis_cache: dict[str, tuple[dict, float]] = None  # hash -> (analysis, timestamp)
    
    def __post_init__(self):
        if self.file_cache is None:
            self.file_cache = {}
        if self.analysis_cache is None:
            self.analysis_cache = {}

#Caching Utility

def get_file_hash(content: str) -> str:
    """Generate hash for file content for caching"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def is_cache_valid(timestamp: float, max_age_seconds: int = 300) -> bool:
    """Check if cache entry is still valid (default 5 minutes)"""
    return (datetime.now().timestamp() - timestamp) < max_age_seconds

@functools.lru_cache(maxsize=128)
def get_file_language_cached(file_path: str) -> str:
    """Cached version of get_file_language"""
    return get_file_language(file_path)



@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Initialize server with project context"""
    project_root = Path.cwd()
    allowed_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb'}
    
    context = ServerContext(
        project_root=project_root,
        allowed_extensions=allowed_extensions
    )
    
    print(f"Code Debug Agent initialized. Project root: {project_root}")
    yield context


# Create MCP server with lifespan
mcp = FastMCP("Code Debug Agent", lifespan=server_lifespan)


def is_safe_path(file_path: str, project_root: Path) -> bool:
    """Check if file path is safe and within project boundaries"""
    try:
        full_path = (project_root / file_path).resolve()
        return str(full_path).startswith(str(project_root))
    except Exception:
        return False


def get_file_language(file_path: str) -> str:
    """Determine programming language from file extension"""
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


@mcp.resource("code://{file_path}")
def read_code_file(file_path: str) -> str:
    """Read a code file for review and analysis with caching"""
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        raise ValueError(f"Access denied: {file_path} is outside project directory")
    
    full_path = server_ctx.project_root / file_path
    
    if not full_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if full_path.suffix not in server_ctx.allowed_extensions:
        raise ValueError(f"Unsupported file type: {full_path.suffix}")
    
    try:
        # Check cache first
        file_stat = full_path.stat()
        file_mtime = file_stat.st_mtime
        
        if (file_path in server_ctx.file_cache and 
            server_ctx.file_cache[file_path][1] >= file_mtime):
            # Cache hit - return cached content
            cached_content, _, _ = server_ctx.file_cache[file_path]
            language = get_file_language_cached(file_path)
            
            return f"""# File: {file_path} (cached)
# Language: {language}
# Size: {len(cached_content)} characters
# Lines: {len(cached_content.splitlines())}

```{language}
{cached_content}
```
"""
        
        # Cache miss - read file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update cache
        content_hash = get_file_hash(content)
        server_ctx.file_cache[file_path] = (content, file_mtime, content_hash)
        
        language = get_file_language_cached(file_path)
        
        return f"""# File: {file_path}
# Language: {language}
# Size: {len(content)} characters
# Lines: {len(content.splitlines())}

```{language}
{content}
```
"""
    except Exception as e:
        raise RuntimeError(f"Error reading file {file_path}: {str(e)}")


@mcp.tool()
def review_code(file_path: str, focus_areas: Optional[str] = None) -> str:
    """
    Optimized comprehensive code review tool with caching and improved analysis.
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        return f"Error: Access denied to {file_path}"
    
    full_path = server_ctx.project_root / file_path
    
    if not full_path.exists():
        return f"Error: File {file_path} not found"
    
    try:
        # Read file with caching
        file_stat = full_path.stat()
        file_mtime = file_stat.st_mtime
        
        # Check file cache
        if (file_path in server_ctx.file_cache and 
            server_ctx.file_cache[file_path][1] >= file_mtime):
            content, _, content_hash = server_ctx.file_cache[file_path]
        else:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            content_hash = get_file_hash(content)
            server_ctx.file_cache[file_path] = (content, file_mtime, content_hash)
        
        # Check analysis cache
        cache_key = f"{content_hash}_{focus_areas or 'general'}"
        if (cache_key in server_ctx.analysis_cache and 
            is_cache_valid(server_ctx.analysis_cache[cache_key][1])):
            analysis, _ = server_ctx.analysis_cache[cache_key]
        else:
            # Perform analysis
            analysis = perform_code_analysis(content, file_path, focus_areas)
            server_ctx.analysis_cache[cache_key] = (analysis, datetime.now().timestamp())
        
        # Generate review report (rest of the function remains the same but uses cached analysis)
        language = get_file_language_cached(file_path)
        
        review_text = f"""
# Code Review Report for {file_path}

## File Information
- **Language**: {language}
- **Lines of Code**: {analysis['file_info']['lines']}
- **File Size**: {analysis['file_info']['size_kb']} KB

## Structure Analysis
"""
        
        if analysis['structure_analysis']:
            for key, value in analysis['structure_analysis'].items():
                if isinstance(value, list):
                    review_text += f"- **{key.replace('_', ' ').title()}**: {', '.join(map(str, value)) if value else 'None'}\n"
                else:
                    review_text += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        review_text += "\n## Code Content\n"
        review_text += f"```{language}\n{content}\n```\n"
        
        if analysis['potential_issues']:
            review_text += "\n## Potential Issues\n"
            for issue in analysis['potential_issues']:
                review_text += f"- ⚠️ {issue}\n"
        
        if analysis['suggestions']:
            review_text += "\n## Suggestions\n"
            for suggestion in analysis['suggestions']:
                review_text += f"- 💡 {suggestion}\n"
        
        return review_text
        
    except Exception as e:
        return f"Error during code review: {str(e)}\n{traceback.format_exc()}"
    

def perform_code_analysis(content: str, file_path: str, focus_areas: Optional[str] = None) -> dict:
    """Separate function to perform code analysis for better caching and reuse"""
    lines = content.splitlines()
    language = get_file_language_cached(file_path)
    
    analysis = {
        'file_info': {
            'path': file_path,
            'language': language,
            'lines': len(lines),
            'characters': len(content),
            'size_kb': round(len(content.encode('utf-8')) / 1024, 2)
        },
        'structure_analysis': {},
        'potential_issues': [],
        'suggestions': []
    }
    
    # Python-specific analysis (optimized)
    if language == 'python':
        try:
            tree = ast.parse(content)
            
            # Use list comprehensions for better performance
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
            
            analysis['structure_analysis'] = {
                'functions': len(functions),
                'classes': len(classes),
                'imports': len(imports),
                'function_names': functions,
                'class_names': classes
            }
            
            # Optimized issue detection
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if len(node.body) > 50:
                        analysis['potential_issues'].append(
                            f"Function '{node.name}' is quite long ({len(node.body)} statements)"
                        )
                    if not node.args.args and node.name not in ['__init__', '__new__', '__del__']:
                        analysis['potential_issues'].append(
                            f"Function '{node.name}' takes no parameters - consider if this is intentional"
                        )
                        
        except SyntaxError as e:
            analysis['potential_issues'].append(f"Syntax Error: {str(e)}")
    
    # Optimized general checks using any() for early termination
    if any(line.strip().startswith(('TODO', 'FIXME')) for line in lines):
        analysis['potential_issues'].append("Contains TODO/FIXME comments")
    
    if any(len(line) > 120 for line in lines):
        analysis['potential_issues'].append("Some lines exceed 120 characters")
    
    # Focus area analysis (optimized)
    if focus_areas:
        focus_list = [area.strip().lower() for area in focus_areas.split(',')]
        analysis['focus_areas'] = focus_list
        
        if 'security' in focus_list:
            security_patterns = ['eval(', 'exec(', 'input(', 'raw_input(']
            for pattern in security_patterns:
                if pattern in content:
                    analysis['potential_issues'].append(f"Security concern: Found {pattern}")
        
        if 'performance' in focus_list:
            if 'for' in content and 'append(' in content:
                analysis['suggestions'].append("Consider using list comprehensions for better performance")
    
    return analysis


@mcp.tool()
def write_ai_code(
    file_path: str,
    code_content: str,
    language: str = "python",
    overwrite: bool = False,
    backup: bool = True
) -> str:
    """
    Write AI-generated code to existing or new files with validation.
    
    Args:
        file_path: Path to the target file
        code_content: The AI-generated code content to write
        language: Programming language for validation
        overwrite: Whether to overwrite existing files
        backup: Whether to create backup of existing files
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        return f"Error: Access denied to {file_path}"
    
    full_path = server_ctx.project_root / file_path
    
    try:
        # Handle existing files
        if full_path.exists():
            if not overwrite:
                return f"Error: File {file_path} already exists. Set overwrite=True to replace it."
            
            # Create backup if requested
            if backup:
                with open(full_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                backup_path = full_path.with_suffix(full_path.suffix + '.ai_backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
        
        # Create directory if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate code based on language
        validation_result = ""
        if language.lower() == 'python':
            try:
                ast.parse(code_content)
                validation_result = "✅ Python syntax validation passed"
            except SyntaxError as e:
                return f"❌ Invalid Python syntax: {str(e)}\nPlease fix the code before writing to file."
        else:
            validation_result = f"⚠️ {language} syntax not validated (parser not available)"
        
        # Write the code
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        file_size = len(code_content.encode('utf-8'))
        line_count = len(code_content.splitlines())
        
        return f"""AI-generated code written successfully!

**File**: {file_path}
**Action**: {'Overwritten' if full_path.exists() else 'Created'}
**Language**: {language}
**Size**: {file_size} bytes
**Lines**: {line_count}
**Backup Created**: {'Yes' if backup and full_path.exists() else 'No'}
**Validation**: {validation_result}

**Code Written**:
```{language}
{code_content}
```

**Status**: ✅ AI code successfully written to file
"""
        
    except Exception as e:
        return f"Error writing AI code to {file_path}: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def debug_and_fix_code(
    file_path: str, 
    issue_description: Optional[str] = None,
    proposed_fix: Optional[str] = None,
    auto_analyze: bool = True,
    backup: bool = True
) -> str:
    """
    Enhanced debug tool that can read file contents, analyze issues, and apply fixes.
    
    Args:
        file_path: Path to the code file to debug
        issue_description: Optional description of the issue (if not provided, will auto-analyze)
        proposed_fix: Optional fix code (if not provided, will suggest based on analysis)
        auto_analyze: Whether to automatically analyze the code for issues
        backup: Whether to create a backup before making changes
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        return f"Error: Access denied to {file_path}"
    
    full_path = server_ctx.project_root / file_path
    
    if not full_path.exists():
        return f"Error: File {file_path} not found"
    
    try:
        # Read current file content
        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        language = get_file_language(file_path)
        analysis_result = ""
        detected_issues = []
        
        # Auto-analyze the code if requested
        if auto_analyze:
            analysis_result += f"\n## 🔍 Auto-Analysis of {file_path}\n"
            
            # Basic code analysis
            lines = original_content.splitlines()
            analysis_result += f"- **Lines of code**: {len(lines)}\n"
            analysis_result += f"- **File size**: {len(original_content)} characters\n"
            
            # Python-specific analysis
            if language == 'python':
                try:
                    tree = ast.parse(original_content)
                    analysis_result += "- **Syntax**: ✅ Valid Python syntax\n"
                    
                    # Analyze structure
                    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    
                    analysis_result += f"- **Functions**: {len(functions)}\n"
                    analysis_result += f"- **Classes**: {len(classes)}\n"
                    
                    # Check for common issues
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check for very long functions
                            if len(node.body) > 50:
                                detected_issues.append(f"Function '{node.name}' is very long ({len(node.body)} statements)")
                            
                            # Check for functions without docstrings
                            if not ast.get_docstring(node):
                                detected_issues.append(f"Function '{node.name}' missing docstring")
                        
                        # Check for bare except clauses
                        if isinstance(node, ast.ExceptHandler) and node.type is None:
                            detected_issues.append("Found bare 'except:' clause - should specify exception type")
                    
                except SyntaxError as e:
                    analysis_result += f"- **Syntax**: ❌ Python syntax error: {str(e)}\n"
                    detected_issues.append(f"Syntax Error at line {e.lineno}: {e.msg}")
            
            # General code quality checks
            for i, line in enumerate(lines, 1):
                if len(line) > 120:
                    detected_issues.append(f"Line {i} exceeds 120 characters")
                if line.strip().startswith('TODO') or line.strip().startswith('FIXME'):
                    detected_issues.append(f"Line {i}: Contains TODO/FIXME comment")
                if 'print(' in line and language == 'python':
                    detected_issues.append(f"Line {i}: Debug print statement found")
        
        # Show current file content
        result = f"""# Debug Analysis for {file_path}

## 📄 Current File Content
```{language}
{original_content}
```
{analysis_result}
"""
        
        # Show detected issues
        if detected_issues:
            result += "\n## ⚠️ Detected Issues\n"
            for issue in detected_issues:
                result += f"- {issue}\n"
        else:
            result += "\n## ✅ No Major Issues Detected\n"
        
        # Handle issue description
        if issue_description:
            result += f"\n## 🐛 Reported Issue\n{issue_description}\n"
        
        # Apply fix if provided
        if proposed_fix:
            # Create backup if requested
            if backup:
                backup_path = full_path.with_suffix(full_path.suffix + '.debug_backup')
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
            
            # Validate the fix
            validation_result = ""
            if language == 'python':
                try:
                    ast.parse(proposed_fix)
                    validation_result = "✅ Fix validation passed"
                except SyntaxError as e:
                    result += f"\n## ❌ Fix Validation Failed\n**Error**: {str(e)}\n**Status**: Fix not applied\n"
                    return result
            
            # Apply the fix
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(proposed_fix)
            
            result += f"""
## 🔧 Fix Applied
**Validation**: {validation_result}
**Backup Created**: {'Yes' if backup else 'No'}

**New Code**:
```{language}
{proposed_fix}
```

**Status**: ✅ Fix successfully applied
"""
        else:
            result += f"""
## 🔧 Next Steps
To apply a fix:
1. Analyze the issues above
2. Create corrected code
3. Call this function again with the `proposed_fix` parameter

**Example**:
```python
debug_and_fix_code(
    file_path="{file_path}",
    issue_description="Description of fix",
    proposed_fix="your_corrected_code_here"
)
```
"""
        
        return result
        
    except Exception as e:
        return f"Error during debug analysis: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def modify_code_section(
    file_path: str,
    start_line: int,
    end_line: int,
    new_code: str,
    description: str = "Code modification",
    backup: bool = True
) -> str:
    """
    Modify a specific section of code in a file.
    
    Args:
        file_path: Path to the code file
        start_line: Starting line number (1-based)
        end_line: Ending line number (1-based, inclusive)
        new_code: New code to replace the section
        description: Description of the modification
        backup: Whether to create a backup
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        return f"Error: Access denied to {file_path}"
    
    full_path = server_ctx.project_root / file_path
    
    if not full_path.exists():
        return f"Error: File {file_path} not found"
    
    try:
        # Read current content
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Validate line numbers
        if start_line < 1 or end_line < 1 or start_line > total_lines or end_line > total_lines:
            return f"Error: Invalid line numbers. File has {total_lines} lines."
        
        if start_line > end_line:
            return f"Error: Start line ({start_line}) cannot be greater than end line ({end_line})"
        
        # Create backup if requested
        if backup:
            backup_path = full_path.with_suffix(full_path.suffix + '.modify_backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        
        # Show what's being replaced
        original_section = ''.join(lines[start_line-1:end_line])
        
        # Replace the section
        new_lines = lines[:start_line-1] + [new_code + '\n'] + lines[end_line:]
        
        # Write modified content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        language = get_file_language(file_path)
        
        # Validate if Python
        if language == 'python':
            try:
                new_content = ''.join(new_lines)
                ast.parse(new_content)
                validation = "✅ Python syntax validation passed"
            except SyntaxError as e:
                # Restore original content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                return f"❌ Syntax error in modified code: {str(e)}\nOriginal file restored."
        else:
            validation = f"⚠️ {language} syntax not validated"
        
        return f"""Code section modified successfully!

**File**: {file_path}
**Lines Modified**: {start_line} to {end_line}
**Description**: {description}
**Backup Created**: {'Yes' if backup else 'No'}
**Validation**: {validation}

**Original Code** (lines {start_line}-{end_line}):
```{language}
{original_section.rstrip()}
```

**New Code**:
```{language}
{new_code}
```

**Status**: ✅ Code section successfully modified
"""
        
    except Exception as e:
        return f"Error modifying code section: {str(e)}\n{traceback.format_exc()}"


@mcp.tool()
def clear_cache() -> str:
    """Clear file and analysis caches to free memory"""
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    file_cache_size = len(server_ctx.file_cache)
    analysis_cache_size = len(server_ctx.analysis_cache)
    
    server_ctx.file_cache.clear()
    server_ctx.analysis_cache.clear()
    
    # Clear function cache
    get_file_language_cached.cache_clear()
    
    return f"""Cache cleared successfully!

**File cache entries cleared**: {file_cache_size}
**Analysis cache entries cleared**: {analysis_cache_size}
**Function cache cleared**: Yes

**Status**: ✅ All caches cleared, memory freed
"""



@mcp.tool()
def list_project_files(directory: str = ".", file_extensions: str = "py,js,ts") -> str:
    """
    Optimized version of list_project_files with better performance for large directories.
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(directory, server_ctx.project_root):
        return f"Error: Access denied to {directory}"
    
    try:
        scan_path = server_ctx.project_root / directory
        if not scan_path.exists():
            return f"Error: Directory {directory} not found"
        
        extensions = {f".{ext.strip()}" for ext in file_extensions.split(',')}  # Use set for O(1) lookup
        files = []
        
        # More efficient file collection
        for file_path in scan_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                files.append(file_path)
        
        # Sort and format results
        files.sort()
        result = f"Code files in {directory}:\n\n"
        
        total_size = 0
        for file in files:
            relative_path = file.relative_to(server_ctx.project_root)
            size = file.stat().st_size
            total_size += size
            result += f"- {relative_path} ({size} bytes)\n"
        
        if not files:
            result += "No matching files found.\n"
        else:
            result += f"\nTotal: {len(files)} files, {total_size} bytes\n"
        
        return result
        
    except Exception as e:
        return f"Error listing files: {str(e)}"


@mcp.tool()
def create_code_file(file_path: str, content: str, language: str = "python") -> str:
    """
    Create a new code file with the specified content.
    
    Args:
        file_path: Path for the new file
        content: Code content to write
        language: Programming language (for validation)
    """
    context = mcp.get_context()
    server_ctx = context.request_context.lifespan_context
    
    if not is_safe_path(file_path, server_ctx.project_root):
        return f"Error: Access denied to {file_path}"
    
    full_path = server_ctx.project_root / file_path
    
    if full_path.exists():
        return f"Error: File {file_path} already exists. Use debug_and_fix_code to modify existing files."
    
    try:
        # Create directory if it doesn't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Validate content if it's Python
        if language.lower() == 'python':
            try:
                ast.parse(content)
            except SyntaxError as e:
                return f"Error: Invalid Python syntax - {str(e)}"
        
        # Write the file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"""New code file created successfully!

**File**: {file_path}
**Language**: {language}
**Size**: {len(content)} characters

**Content**:
```{language}
{content}
```

**Status**: ✅ File created and ready for use
"""
        
    except Exception as e:
        return f"Error creating file {file_path}: {str(e)}"


@mcp.prompt()
def code_review_prompt(file_path: str, focus: str = "general") -> str:
    """Generate a comprehensive code review prompt"""
    return f"""Please perform a detailed code review of the file: {file_path}

Focus areas: {focus}

Please analyze:
1. Code structure and organization
2. Potential bugs and issues
3. Performance considerations
4. Security vulnerabilities
5. Best practice adherence
6. Readability and maintainability
7. Testing coverage needs

Provide specific suggestions for improvements with code examples where applicable."""


@mcp.prompt()
def debug_prompt(file_path: str, error_description: str) -> str:
    """Generate a debugging analysis prompt"""
    return f"""Please help debug the following issue in {file_path}:

**Error/Issue Description**: {error_description}

Please:
1. Analyze the code to identify the root cause
2. Explain why the issue occurs
3. Provide a step-by-step fix
4. Suggest improvements to prevent similar issues
5. Include the complete corrected code

Focus on providing a working solution that can be directly applied."""


if __name__ == "__main__":
    # Run the MCP server using stdio transport
    mcp.run(transport="stdio")