import json
import os

ws_dir = 'workspaces/a42b1b867bf8'
setup_output = json.load(open(f'{ws_dir}/setup.json', encoding='utf-8'))
experiment_blueprint = json.load(open(f'{ws_dir}/blueprint.json', encoding='utf-8'))
code_plan = json.load(open(f'{ws_dir}/code_plan.json', encoding='utf-8'))

topic = 'SAPI-CBM'

for spec in code_plan.get('files', []):
    path = spec.get('path', 'unknown')
    user_prompt = f"""
You are implementing the file `{path}`.

### Overall Code Plan (for context):
```json
{json.dumps(code_plan, indent=2)}
```

### File Specification:
{json.dumps(spec, indent=2)}

### Topic and context
{topic}

### Blueprint
```json
{json.dumps(experiment_blueprint, indent=2)}
```

### Setup Output (available resources)
```json
{json.dumps(setup_output, indent=2)}
```
"""
    print(f'File {path}: {len(user_prompt)} characters (~{len(user_prompt)//3} tokens)')
