import os
import argparse

def delete_files_with_backslashes(root_dir):
    deleted_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if '\\' in fname:
                full_path = os.path.join(dirpath, fname)
                try:
                    os.remove(full_path)
                    deleted_files.append(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")
    if not deleted_files:
        print("No files with backslashes found.")
    else:
        print(f"\nTotal deleted: {len(deleted_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete files with backslashes in their names.")
    parser.add_argument("directory", help="Path to the root directory to scan")
    args = parser.parse_args()

    delete_files_with_backslashes(args.directory)