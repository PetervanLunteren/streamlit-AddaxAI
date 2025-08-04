# Developer Guidelines - AddaxAI Streamlit

This document outlines the architectural decisions and coding standards for the AddaxAI Streamlit application. Please follow these guidelines to maintain consistency across the codebase.

## Variable Storage Architecture

The application uses a **three-tier variable storage system** to ensure proper data persistence, session management, and crash recovery. All variables **must** be stored according to these rules:

### **Category 1: Temporary Variables (Session State)**
**Storage**: `st.session_state["tool_name"]["variable_name"]`  
**Purpose**: Variables that should only exist during the current user session

**Examples**:
```python
# Tool-specific temporary variables
st.session_state["analyse_advanced"]["step"]                # Current stepper position
st.session_state["analyse_advanced"]["selected_folder"]     # Currently browsed folder (while selecting)
st.session_state["analyse_advanced"]["selected_species"]    # Current species selection
st.session_state["analyse_advanced"]["required_env_name"]   # Modal state variables
st.session_state["analyse_advanced"]["download_modelID"]    # Modal state variables

# Cross-tool temporary variables  
st.session_state["shared"]["map_interaction_data"]          # Map interaction state
st.session_state["shared"]["current_tab"]                   # UI navigation state
```

**When to use**:
- UI state (current step, widget selections, modal states)
- Temporary user inputs while configuring
- Any data that can be safely lost if the session crashes
- Data that should reset when user starts over

**Helper Functions**:
```python
from utils.common import init_session_state, get_session_var, set_session_var, update_session_vars, clear_vars

# Initialize session state for your tool
init_session_state("your_tool_name")

# Get/set individual variables
value = get_session_var("your_tool_name", "variable_name", default_value)
set_session_var("your_tool_name", "variable_name", value)

# Update multiple variables at once
update_session_vars("your_tool_name", {
    "var1": "value1",
    "var2": "value2"
})

# Clear all temporary variables for your tool
clear_vars("your_tool_name")
```

### **Category 2: Persistent Variables (JSON Files)**
**Storage**: `config/tool_name.json`  
**Purpose**: Variables that must survive session crashes and app restarts

**Examples**:
```python
# config/analyse_advanced.json
{
    "process_queue": [
        {
            "selected_folder": "/path/to/deployment",
            "selected_projectID": "kenya_project",
            "selected_locationID": "camera_01",
            "selected_min_datetime": "2024-01-01T10:00:00",
            "selected_det_modelID": "MD5A-0-0",
            "selected_cls_modelID": "SAH-DRY-ADS-v1",
            "selected_species": ["lion", "elephant", "zebra"]
        }
    ]
}

# config/general_settings.json
{
    "lang": "en",
    "mode": 1,
    "selected_projectID": "active_project_id"
}
```

**When to use**:
- Processing queues (critical for crash recovery)
- User preferences that should persist across sessions
- Committed/finalized selections (after user clicks "Add to Queue")
- Any data that would be catastrophic to lose

**Helper Functions**:
```python
from utils.common import load_vars, update_vars, replace_vars

# Load variables from persistent storage
vars_data = load_vars("tool_name")
process_queue = vars_data.get("process_queue", [])

# Update specific variables
update_vars("tool_name", {"new_key": "new_value"})

# Replace entire variable set
replace_vars("tool_name", {"completely": "new_data"})
```

### **Category 3: Global Variables (User Config)**
**Storage**: `~/.config/AddaxAI/map.json` (user config directory)  
**Purpose**: Data that survives app version updates and uninstalls

**Examples**:
```python
{
    "projects": {
        "kenya_wildlife_2024": {
            "name": "Kenya Wildlife Survey 2024",
            "created_date": "2024-01-15",
            "locations": {
                "camera_001": {
                    "name": "Watering Hole North",
                    "latitude": -1.2921,
                    "longitude": 36.8219,
                    "deployments": [...]
                }
            }
        }
    }
}
```

**When to use**:
- Project definitions and metadata
- Camera locations and deployment history
- Long-term user preferences
- Any data that should survive app version upgrades

**Helper Functions**:
```python
from utils.common import load_map

# Load global settings
settings, settings_file = load_map()
projects = settings.get("projects", {})
```

## Variable Lifecycle Management

### **Selection → Commitment Pattern**
Variables follow a lifecycle from temporary to persistent:

1. **Selection Phase**: User makes choices → stored in session state
2. **Commitment Phase**: User confirms choices → data moves to persistent storage
3. **Processing Phase**: Data remains in persistent storage for crash recovery

**Example**:
```python
# Step 1: User selects folder (temporary)
selected_folder = browse_directory_widget()
set_session_var("analyse_advanced", "selected_folder", selected_folder)

# Step 2: User selects project details (temporary)
update_session_vars("analyse_advanced", {
    "selected_projectID": project_id,
    "selected_locationID": location_id
})

# Step 3: User commits to queue (persistent)
def add_deployment_to_queue():
    # Get temporary selections
    folder = get_session_var("analyse_advanced", "selected_folder")
    project = get_session_var("analyse_advanced", "selected_projectID")
    
    # Create persistent deployment record
    deployment = {
        "selected_folder": folder,
        "selected_projectID": project,
        # ... other committed data
    }
    
    # Save to persistent storage
    queue = load_vars("analyse_advanced").get("process_queue", [])
    queue.append(deployment)
    update_vars("analyse_advanced", {"process_queue": queue})
```

## Tool Development Guidelines

### **1. Tool Initialization**
Every tool should initialize its session state:

```python
from utils.common import init_session_state

# At the top of your tool file
init_session_state("your_tool_name")
```

### **2. State Management**
Use consistent patterns for managing tool state:

```python
# Get current step
step = get_session_var("your_tool_name", "step", 0)

# Advance to next step with data
update_session_vars("your_tool_name", {
    "step": step + 1,
    "user_selection": selected_value
})

# Clear all temporary data (preserves persistent data)
clear_vars("your_tool_name")
```

### **3. Cross-Tool Communication**
For variables needed across multiple tools:

```python
# Temporary cross-tool data
st.session_state["shared"]["variable_name"] = value

# Persistent cross-tool data (rare - use general_settings)
update_vars("general_settings", {"shared_preference": value})
```

### **4. Error Handling**
Always provide defaults and handle missing data:

```python
# Session state with defaults
current_step = get_session_var("tool_name", "step", 0)
user_input = get_session_var("tool_name", "user_input", "")

# Persistent storage with defaults
vars_data = load_vars("tool_name")
queue = vars_data.get("process_queue", [])
settings = vars_data.get("settings", {})
```

## Implementation Lessons Learned

### **Session State Organization Issues**
During development, we discovered several critical issues with session state management:

**Problem 1: Widget-Created Variables in Top-Level Session State**
```python
# Streamlit widgets automatically create top-level session state variables
# This breaks our structured organization:
st.selectbox("Project", options=[...], key="project_selection")
# ❌ Creates: st.session_state["project_selection"]  
# ✅ Should be: st.session_state["shared"]["project_selection"]
```

**Solution: Cleanup Pattern**
```python
# After widget renders, move to structured session state
if "project_selection" in st.session_state:
    set_session_var("shared", "project_selection", st.session_state["project_selection"])
    del st.session_state["project_selection"]
```

**Problem 2: Species Selector State Management**
The species selector widget had complex state (selected_nodes, expanded_nodes, last_selected) that wasn't properly structured:

```python
# ❌ Before: Variables scattered in top-level session state
st.session_state["selected_nodes"] = [...]
st.session_state["expanded_nodes"] = [...]

# ✅ After: Properly structured
selected_nodes = get_session_var("analyse_advanced", "selected_nodes", [])
expanded_nodes = get_session_var("analyse_advanced", "expanded_nodes", [])
```

**Problem 3: DateTime Storage Type Mismatch**
```python
# ❌ Inconsistent storage caused TypeError: fromisoformat: argument must be str
if isinstance(selected_min_datetime, str):
    selected_min_datetime = datetime.fromisoformat(selected_min_datetime)

# ✅ Always store as ISO string for consistency
set_session_var("analyse_advanced", "selected_min_datetime", 
                selected_min_datetime.isoformat())
```

**Problem 4: Redundant Storage of Persistent Variables**
Originally, `lang`, `mode`, and `selected_projectID` were stored in both session state and persistent storage:

```python
# ❌ Before: Redundant storage
st.session_state["shared"]["mode_selection"] = mode  # Unnecessary
vars["general_settings"]["mode"] = mode              # Correct

# ✅ After: Only persistent storage
def on_mode_change():
    update_vars("general_settings", {"mode": st.session_state["mode_selection"]})
    # No session state storage needed
```

### **Defensive Programming for Callbacks**
Widget callbacks can fail if session state variables are deleted:

```python
# ❌ Fragile callback
def on_project_change():
    project = st.session_state["project_selection"]  # KeyError if deleted
    
# ✅ Defensive callback  
def on_project_change():
    if "project_selection" in st.session_state:
        project = st.session_state["project_selection"]
        # ... handle change
```

## Best Practices

### **✅ Do**
- Always initialize session state for your tool
- Use descriptive variable names
- Provide sensible defaults for all variables
- Clear temporary state when user starts over
- Store critical data (queues, preferences) persistently
- Test your tool's behavior after session reset
- Use cleanup patterns for widget-created session state variables
- Store datetime objects as ISO strings for consistency
- Make widget callbacks defensive against missing session state
- Only store variables in session state if they're truly temporary

### **❌ Don't**
- Store temporary UI state in persistent files
- Store critical processing data only in session state
- Access session state directly - use helper functions
- Forget to handle missing/default values
- Mix temporary and persistent data in the same storage
- Store persistent preferences in session state (causes redundancy)
- Assume session state variables will always exist in callbacks
- Store different data types for the same variable (datetime vs string)

### **🔧 Migration Lessons**
When updating existing code:
- Don't migrate existing data - just change where new data is stored
- Gradually update tools to use the new pattern
- Test thoroughly after changes
- Pay special attention to widget callback functions
- Check for variables being stored in multiple places (session state + persistent)
- Test with page refreshes to ensure session state organization works
- Use browser dev tools to inspect actual session state structure during debugging

## File Structure

```
streamlit-AddaxAI/
├── pages/                          # Individual page implementations
│   ├── analysis_advanced.py        # Example of properly implemented page
│   └── other_page.py
├── utils/
│   ├── common.py                    # Session state helper functions
│   └── tool_specific_utils.py
├── config/                          # Persistent storage (git-ignored)
│   ├── analysis_advanced.json
│   ├── general_settings.json
│   └── other_tool.json
└── ~/.config/AddaxAI/              # Global user data
    └── map.json
```

## Testing Your Implementation

Verify your variable storage is working correctly:

```python
# Test session state persistence
init_session_state("your_tool")
set_session_var("your_tool", "test_var", "test_value")
assert get_session_var("your_tool", "test_var") == "test_value"

# Test clear functionality
clear_vars("your_tool")
assert get_session_var("your_tool", "test_var", "default") == "default"

# Test persistent storage
update_vars("your_tool", {"persistent_key": "persistent_value"})
data = load_vars("your_tool")
assert data["persistent_key"] == "persistent_value"
```

## Questions?

If you're unsure about where to store a variable:

1. **Will the user be upset if this data disappears when they refresh the page?**
   - Yes → Persistent storage
   - No → Session state

2. **Is this data critical for crash recovery?**
   - Yes → Persistent storage
   - No → Session state

3. **Should this data survive app version updates?**
   - Yes → Global storage (map.json)
   - No → Persistent storage (vars/*.json)

4. **Is this just UI state or temporary user input?**
   - Yes → Session state
   - No → Consider persistent storage

---

**Remember**: The goal is crash recovery, user experience, and maintainable code. When in doubt, err on the side of persistence for important data and session state for UI/temporary data.