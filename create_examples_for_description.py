import json
import os
import re

def is_empty(value):
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False

def merge_values(existing, new_value):
    if isinstance(existing, list):
        if new_value not in existing:
            existing.append(new_value)
        return existing
    return [existing, new_value] if existing != new_value else [existing]

def merge_structures(base, new):
    if isinstance(base, dict) and isinstance(new, dict):
        for key, value in new.items():
            if key in base:
                base[key] = merge_structures(base[key], value)
            else:
                base[key] = value
    elif isinstance(base, list) and isinstance(new, list):
        if base and isinstance(base[0], dict) and new and isinstance(new[0], dict):
            base_item = base[0]
            for k, v in new[0].items():
                if k in base_item:
                    base_item[k] = merge_structures(base_item[k], v)
                else:
                    base_item[k] = v
        else:
            for item in new:
                if item not in base:
                    base.append(item)
    else:
        base = merge_values(base, new)
    return base

def build_structure_with_values(data):
    if isinstance(data, dict):
        output = {}
        for k, v in data.items():
            result = build_structure_with_values(v)
            output[k] = result
        return output
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            merged = {}
            for item in data:
                item_struct = build_structure_with_values(item)
                for key, value in item_struct.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        merged[key] = merge_structures(merged[key], value)
            return [merged]
        else:
            cleaned = [item for item in data if not is_empty(item)]
            return sorted(set(cleaned), key=str.lower) if cleaned else []
    else:
        return [data] if not is_empty(data) else []

def inline_small_arrays(json_string, max_items=20):
    # Matches small arrays of strings and condenses them into one line
    pattern = re.compile(r'\[\s*((?:"[^"]*"\s*,\s*){1,' + str(max_items - 1) + r'}"[^"]*")\s*\]', re.MULTILINE)
    return pattern.sub(lambda m: "[{}]".format(", ".join(i.strip() for i in m.group(1).split(","))), json_string)

def process_json_folder(input_directory="ground-truth", output_path="final_output_with_values.json"):
    if not os.path.exists(input_directory):
        print(f"❌ Error: Directory '{input_directory}' not found.")
        return

    all_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]
    if not all_files:
        print(f"⚠️ No JSON files found in '{input_directory}'.")
        return

    final_structure = None

    for file in all_files:
        path = os.path.join(input_directory, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
                structure_with_values = build_structure_with_values(input_data)

                if final_structure is None:
                    final_structure = structure_with_values
                else:
                    final_structure = merge_structures(final_structure, structure_with_values)

        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

    if final_structure is not None:
        raw_json = json.dumps(final_structure, indent=2, ensure_ascii=False)
        formatted_json = inline_small_arrays(raw_json)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_json)
        print(f"✅ Final output with values saved to {output_path}")

if __name__ == "__main__":
    process_json_folder()
