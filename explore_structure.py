import json

# Load the JSON file
json_file = 'ENdiele20.json'

with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 80)
print(f"Loaded JSON file: {json_file}")
print("=" * 80)
print("\n")

# Show the type and basic structure
print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
print(f"Number of values: {len(data)}")
print("\n")

# Show keys and values structure (without printing actual data)
print("=" * 80)
print("Dictionary Structure (Keys and Value Types):")
print("=" * 80)
print("\n")

def print_structure(obj, indent=0, max_depth=10):
    """Recursively print the structure of a dictionary/list without printing actual values"""
    if indent > max_depth:
        print("  " * indent + "... (max depth reached)")
        return
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            value_type = type(value).__name__
            if isinstance(value, dict):
                print("  " * indent + f"'{key}': dict ({len(value)} keys)")
                print_structure(value, indent + 1, max_depth)
            elif isinstance(value, list):
                print("  " * indent + f"'{key}': list ({len(value)} items)")
                if len(value) > 0:
                    print("  " * indent + f"  -> Item type: {type(value[0]).__name__}")
                    if isinstance(value[0], (dict, list)):
                        print("  " * indent + f"  -> First item structure:")
                        print_structure(value[0], indent + 2, max_depth)
            else:
                print("  " * indent + f"'{key}': {value_type}")
    elif isinstance(obj, list):
        if len(obj) > 0:
            print("  " * indent + f"List items are: {type(obj[0]).__name__}")
            if isinstance(obj[0], (dict, list)):
                print("  " * indent + f"  -> First item structure:")
                print_structure(obj[0], indent + 1, max_depth)

print_structure(data)

