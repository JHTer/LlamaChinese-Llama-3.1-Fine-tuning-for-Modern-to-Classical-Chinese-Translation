import os
import pandas as pd
from typing import List, Tuple

def read_file_content(file_path: str) -> List[str]:
    """
    Reads the content of a file and returns it as a list of lines.

    Args:
        file_path (str): Path to the file to be read.

    Returns:
        List[str]: Content of the file as a list of lines.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().split("\n")
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def process_subfolder(subfolder_path: str) -> List[Tuple[str, str]]:
    """
    Processes a single subfolder, reading all source and target files.

    Args:
        subfolder_path (str): Path to the subfolder to process.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing (source_content, target_content) for each line.
    """
    source_content = read_file_content(os.path.join(subfolder_path, "source.txt"))
    target_content = read_file_content(os.path.join(subfolder_path, "target.txt"))

    if len(source_content) != len(target_content):
        print(f"Warning: Mismatch in line count for {subfolder_path}. Skipping this subfolder.")
        return []

    return list(zip(source_content, target_content))

def get_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    Reads all source and target files from subfolders and combines them into a dataset.

    Args:
        folder_path (str): Path to the main folder containing subfolders with source and target files.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing (source_content, target_content) for each line.
    """
    dataset = []

    for root, dirs, files in os.walk(folder_path):
        if "source.txt" in files and "target.txt" in files:
            subfolder_data = process_subfolder(root)
            dataset.extend(subfolder_data)
            print(f"Processed files in {root}")

    return dataset

def main():
    folder_path = "双语数据"
    dataset = get_files(folder_path)

    # Create DataFrame
    df = pd.DataFrame(dataset, columns=["output", "input"])
    df["instruction"] = "请把现代汉语翻译成古文"

    # Print dataset length
    print(f"Dataset length: {len(df)}")

    # Save the dataset into a jsonl file
    output_file = "dataset.jsonl"
    df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    main()