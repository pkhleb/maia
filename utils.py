from config import MODEL, HISTORY_FILE, N_RECENT, SYSTEM_PROMPT, SUMMARY_EVERY_N
import os
from openai_client import OpenAIClient
import ast
from pydantic import BaseModel

class RelevantFiles(BaseModel):
    relevant_files: list[str]

file_structure_dict = {}

def parse_python_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.splitlines()
    tree = ast.parse(content)
    imported_modules = []
    function_definitions = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_modules.append(f"{node.module}.{alias.name}" if node.level == 0 else "." * node.level + node.module)
        elif isinstance(node, ast.FunctionDef):
            calls = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.append(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.append(child.func.attr)
            function_definitions[node.name] = {
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'calls': calls,
                'body': '\n'.join(lines[node.lineno - 1:node.end_lineno])
            }
    return imported_modules, function_definitions

def build_file_structure_dict() -> dict:
    file_structure = {}
    cwd = os.getcwd()
    SKIP_DIRS = {'__pycache__', '.git', '.venv', 'node_modules'}
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
        for file_name in sorted(files):
            if not file_name.endswith('.py'):
                continue
            fpath = os.path.join(root, file_name)
            relative_path = os.path.relpath(fpath, start=cwd)
            try:
                imported_modules, function_definitions = parse_python_file(fpath)
            except Exception as e:
                imported_modules, function_definitions = [], {}
            file_structure[relative_path] = {
                "fpath": fpath,
                "imported_modules": imported_modules,
                "function_definitions": function_definitions
            }
    return file_structure

def build_reverse_import_map(file_structure_dict: dict) -> dict:
    # Maps each module/file to the files that import it
    reverse_map = {}
    for fname, fdata in file_structure_dict.items():
        for mod in fdata['imported_modules']:
            if mod not in reverse_map:
                reverse_map[mod] = []
            reverse_map[mod].append(fname)
    return reverse_map

def get_entry_points(user_input: str, file_structure_dict: dict) -> list:
    entry_points = []

    # Layer 1: explicit function mentions
    for fname, fdata in file_structure_dict.items():
        for func_name in fdata['function_definitions']:
            if func_name in user_input:
                entry_points.append((fname, func_name))

    # Layer 2: explicit file mentions — all functions in that file
    if not entry_points:
        for fname in file_structure_dict:
            if fname in user_input or os.path.basename(fname) in user_input:
                for func_name in file_structure_dict[fname]['function_definitions']:
                    entry_points.append((fname, func_name))

    return entry_points

def collect_context_from_entry_points(entry_points: list, file_structure_dict: dict, reverse_map: dict) -> dict:
    # Returns dict of {relative_path: [function_body, ...]} for all relevant functions
    # For entry point files: include full file
    # For dependencies: include only called functions
    context = {}  # relative_path -> list of bodies to include

    for fname, func_name in entry_points:
        # Include full file for entry point
        if fname not in context:
            context[fname] = 'full'

        # Walk outward: functions this entry point calls
        func_data = file_structure_dict[fname]['function_definitions'].get(func_name, {})
        calls = func_data.get('calls', [])

        for dep_fname, dep_fdata in file_structure_dict.items():
            if dep_fname == fname:
                continue
            for called_func in calls:
                if called_func in dep_fdata['function_definitions']:
                    if dep_fname not in context:
                        context[dep_fname] = []
                    if context[dep_fname] != 'full':
                        body = dep_fdata['function_definitions'][called_func]['body']
                        context[dep_fname].append(body)

        # Walk inward: files that import this file
        base = os.path.splitext(os.path.basename(fname))[0]
        importers = reverse_map.get(base, [])
        for importer in importers:
            if importer not in context:
                context[importer] = []

    return context

def llm_fallback_entry_points(user_input: str, summary: str, file_structure_dict: dict, client: OpenAIClient) -> list:
    # Used when layers 1 and 2 find nothing — ask LLM to identify relevant files
    file_summary = {k: list(v["function_definitions"].keys()) for k, v in file_structure_dict.items()}
    response = client.client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
Files in project (relative path: list of functions):
{file_summary}

Conversation summary: {summary or "None"}

User prompt: {user_input}

Return the most relevant file paths as a JSON list.
"""}
        ],
        extra_body={"guided_json": RelevantFiles.model_json_schema()}
    )
    import json
    raw = response.choices[0].message.content
    raw = raw.strip().removeprefix('```json').removeprefix('```')
    bracket_end = raw.find(']')
    if bracket_end != -1:
        raw = raw[:bracket_end + 1]
    print(f"[LLM fallback raw response: '{raw}']")
    result = json.loads(raw)
    if isinstance(result, list):
        relevant = result
    else:
        relevant = result.get("relevant_files", [])
    # Convert to entry point format: all functions in each relevant file
    entry_points = []
    for rel_path in relevant:
        if rel_path in file_structure_dict:
            for func_name in file_structure_dict[rel_path]['function_definitions']:
                entry_points.append((rel_path, func_name))
    return entry_points

def build_project_context(summary: str, user_input: str, client: OpenAIClient) -> str:
    global file_structure_dict
    if not file_structure_dict:
        file_structure_dict = build_file_structure_dict()

    reverse_map = build_reverse_import_map(file_structure_dict)
    entry_points = get_entry_points(user_input, file_structure_dict)

    # Fall back to LLM if nothing found deterministically
    if not entry_points:
        entry_points = llm_fallback_entry_points(user_input, summary, file_structure_dict, client)

    if not entry_points:
        return ""

    context_map = collect_context_from_entry_points(entry_points, file_structure_dict, reverse_map)

    context = "Here is the relevant project context:\n\n"
    for rel_path, content in context_map.items():
        fpath = file_structure_dict[rel_path]['fpath']
        context += f"=== {rel_path} ===\n"
        if content == 'full':
            try:
                with open(fpath, 'r') as f:
                    context += f.read()
            except Exception as e:
                context += f"[Could not read file: {e}]"
        else:
            context += '\n\n'.join(content)
        context += "\n\n"

    return context

def build_system_prompt(summary: str, user_input: str, client: OpenAIClient) -> str:
    if not summary:
        summary = ""
    project_context = build_project_context(summary, user_input, client)
    prompt = SYSTEM_PROMPT + "\n\n" + project_context
    if summary:
        prompt += "\n\nSummary of earlier conversation:\n" + summary
    return prompt

def maybe_summarize(history: list, current_summary: str, client: OpenAIClient) -> str:
    if len(history) % SUMMARY_EVERY_N != 0:
        return current_summary

    older = history[:-N_RECENT]
    if not older:
        return current_summary

    summary_prompt = "Summarize the following conversation history concisely, capturing key decisions, context, and progress:\n\n"
    for msg in older:
        summary_prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

    try:
        response = client.chat(MODEL, [{"role": "user", "content": summary_prompt}])
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error summarizing conversation: {e}")
        return current_summary
