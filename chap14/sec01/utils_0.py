import os
import json

def save_state(current_path, state):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    
    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    
    with open(f"{current_path}/data/state.json", "w", encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)
