import os
import sys

def rename_folders_remove_after_last_underscore(directory_path, dry_run=True):
    """
    Rename folders by removing all text after the last underscore.
    
    Args:
        directory_path: Path to the directory containing folders to rename
        dry_run: If True, only print what would be done without actually renaming
    """
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a directory.")
        return
    
    # Get all items in the directory
    items = os.listdir(directory_path)
    
    # Filter only directories
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    
    if not folders:
        print("No folders found in the specified directory.")
        return
    
    print(f"Found {len(folders)} folder(s) to process:")
    print("-" * 50)
    
    renamed_count = 0
    skipped_count = 0
    
    for folder in folders:
        old_path = os.path.join(directory_path, folder)
        
        # Find the last underscore
        last_underscore_index = folder.rfind('_')
        
        if last_underscore_index == -1:
            # No underscore found, skip this folder
            print(f" ⏭️ SKIP: '{folder}' (no underscore found)")
            skipped_count += 1
            continue
        
        # Create new folder name (remove everything after last underscore)
        new_name = folder[:last_underscore_index]
        new_path = os.path.join(directory_path, new_name)
        
        # Check if target name already exists
        if os.path.exists(new_path):
            print(f" ⚠️ SKIP: '{folder}' -> '{new_name}' (target already exists)")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f" 🔄 DRY RUN: '{folder}' -> '{new_name}'")
        else:
            try:
                os.rename(old_path, new_path)
                print(f" ✅ RENAMED: '{folder}' -> '{new_name}'")
                renamed_count += 1
            except Exception as e:
                print(f" ❌ ERROR: Failed to rename '{folder}': {e}")
                skipped_count += 1
    
    print("-" * 50)
    print(f"Summary:")
    if dry_run:
        print(f" DRY RUN: Would rename {renamed_count} folder(s), skip {skipped_count} folder(s)")
        print(f" Run with --execute to actually perform the renaming.")
    else:
        print(f" Renamed: {renamed_count} folder(s)")
        print(f" Skipped: {skipped_count} folder(s)")
    
    return renamed_count, skipped_count


def main():
    """Main function to handle command line arguments."""
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python rename_folders.py <directory_path> [--execute]")
        print("\nOptions:")
        print(" --execute Actually rename the folders (without this, it's a dry run)")
        print("\nExample:")
        print(" python rename_folders.py /path/to/folders")
        print(" python rename_folders.py /path/to/folders --execute")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    # Check if --execute flag is present
    dry_run = True
    if len(sys.argv) > 2 and sys.argv[2] == "--execute":
        dry_run = False
    
    # Run the renaming function
    rename_folders_remove_after_last_underscore(directory_path, dry_run)


if __name__ == "__main__":
    main()
