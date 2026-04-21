


import json
import sys
from pathlib import Path


def remove_key(obj, key_to_remove):
    """
    Recursively remove all occurrences of key_to_remove from a JSON object.
    """
    if isinstance(obj, dict):
        return {
            k: remove_key(v, key_to_remove)
            for k, v in obj.items()
            if k != key_to_remove
        }
    elif isinstance(obj, list):
        return [remove_key(item, key_to_remove) for item in obj]
    else:
        return obj


def main(input_file, output_file, key_to_remove):
    # Read JSON
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove specified key recursively
    reduced_data = remove_key(data, key_to_remove)

    # Write reduced JSON (compact format for smaller file size)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(reduced_data, f, separators=(",", ":"))

    print(f"Removed key '{key_to_remove}' and saved reduced JSON to {output_file}")
    print("input file: {} - size: {}".format(input_file, get_file_size(input_file)))
    print("output file: {} - size: {}".format(output_file, get_file_size(output_file)))



def human_readable_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

def get_file_size(filename):
    file_path = Path(filename)
    size_bytes = file_path.stat().st_size

    return human_readable_size(size_bytes)


if __name__ == "__main__":
    # # inputs
    # if len(sys.argv) != 4:
    #     print("Usage: python script.py input.json output.json key_to_remove")
    #     sys.exit(1)

    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    # key_name = sys.argv[3]

    # set inputs
    dir_files = "python-flask-server/test/data"
    input_path = "{}/alzheimer.json".format(dir_files)
    output_path = "{}/alzheimer_reduced.json".format(dir_files)
    key_name = 'attributes'


    input_path = "/home/javaprog/Code/TranslatorWorkspace/UiSummarizerPythonLLM/data/cad.json".format(dir_files)
    output_path = "{}/cad_reduced.json".format(dir_files)

    # run
    main(input_path, output_path, key_name)

    
    