"""
Script to run grade command for multiple competitions and save results in a table.
Finds all potential submission files in competition folders and grades each one.
Excludes sample submission files from grading.
"""

import subprocess
import json
import re
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import argparse
from datetime import datetime
import os


def find_all_submission_files(run_folder: Path, competition_id: str, verbose: bool = False) -> List[Tuple[Path, str]]:
    """
    Find all potential submission files for a given competition in the run folder.
    
    Searches recursively through the entire competition folder for:
    - CSV files with 'submission' in the name
    - Excludes files with 'sample' in the name (sample submissions)
    
    Returns a list of tuples (path, comment) for each found file.
    Comment describes where and how the file was found.
    """
    comp_folder = run_folder / competition_id
    
    if not comp_folder.exists():
        return []
    
    found_files = []
    skipped_samples = []
    
    # Search recursively for CSV files with 'submission' in the name
    for csv_file in comp_folder.rglob("*.csv"):
        if 'submission' in csv_file.name.lower():
            # Skip sample submissions
            if 'sample' in csv_file.name.lower():
                skipped_samples.append(csv_file.name)
                if verbose:
                    print(f"    Skipping sample submission: {csv_file.name}")
                continue
            
            # Generate comment about the file
            comment = []
            
            # Check if it's in a subfolder
            if csv_file.parent != comp_folder:
                rel_path = csv_file.parent.relative_to(comp_folder)
                comment.append(f"Found in {rel_path} folder")
            
            # Check for special naming patterns
            name = csv_file.name.lower()
            if 'baseline' in name:
                comment.append("baseline submission")
            elif name != 'submission.csv':
                comment.append(f"named {csv_file.name}")
            
            # If no specific comment, it's a standard submission
            if not comment:
                comment.append("standard submission in root")
            
            # Add file size info if relevant
            size_kb = csv_file.stat().st_size / 1024
            if size_kb > 1024:  # > 1MB
                comment.append(f"size: {size_kb:.1f}KB")
            
            found_files.append((csv_file.absolute(), ", ".join(comment)))
    
    # Report skipped samples if any
    if skipped_samples and not verbose:
        # We'll report this in the main function
        pass
    
    # Sort by path for consistent ordering
    found_files.sort(key=lambda x: str(x[0]))
    
    return found_files, skipped_samples


def parse_grade_output(output: str, competition_id: str) -> Dict:
    """
    Parse the output from the grade command.
    
    Returns a dictionary with competition_id, metric, and comment.
    """
    result = {
        "competition_id": competition_id,
        "metric": None,
        "comment": None,
        "success": False,
        "submission_path": None
    }
    
    # Try to find JSON data in the output
    json_match = re.search(r'\{[^{}]*"(?:score|competition_id)"[^{}]*\}', output, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Extract submission path if available
            if 'submission_path' in data:
                result["submission_path"] = data['submission_path']
            
            # Check if there was an error based on valid_submission
            if data.get('valid_submission', False) and data.get('score') is not None:
                result["metric"] = data.get('score')
                result["comment"] = "Success"
                result["success"] = True
            else:
                # Look for error messages before the JSON
                error_match = re.search(r'Invalid submission: (.*?)(?:\n|$)', output)
                if error_match:
                    result["comment"] = error_match.group(1)
                else:
                    result["comment"] = "Invalid submission"
                    
        except json.JSONDecodeError:
            # If JSON parsing fails, look for error messages
            error_match = re.search(r'Invalid submission: (.*?)(?:\n|$)', output)
            if error_match:
                result["comment"] = error_match.group(1)
            else:
                result["comment"] = "Failed to parse output"
    else:
        # No JSON found, look for error messages
        error_match = re.search(r'Invalid submission: (.*?)(?:\n|$)', output)
        if error_match:
            result["comment"] = error_match.group(1)
        elif output.strip():
            result["comment"] = output.strip()
        else:
            result["comment"] = "No output from command"
    
    return result


def run_grade_command(competition_id: str, submission_path: Path, verbose: bool = False) -> Tuple[Dict, bool]:
    """
    Run the grade command for a single competition.
    
    Requires submission_path to be provided.
    
    Returns a tuple of (result_dict, success_flag).
    """
    try:
        cmd = ["mlebench", "grade-sample", str(submission_path), competition_id]
        
        if verbose:
            print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Combine stdout and stderr
        output = result.stdout + result.stderr
        
        # Parse the output
        parsed_result = parse_grade_output(output, competition_id)
        
        # Add submission path to result
        parsed_result["submission_path"] = str(submission_path)
        
        # If command failed but we didn't capture an error message
        if result.returncode != 0 and parsed_result["comment"] is None:
            parsed_result["comment"] = f"Command failed with exit code {result.returncode}"
        
        return parsed_result, result.returncode == 0 and parsed_result["success"]
        
    except FileNotFoundError:
        return {
            "competition_id": competition_id,
            "metric": None,
            "comment": "grade command not found. Please ensure it's installed and in PATH",
            "success": False,
            "submission_path": str(submission_path)
        }, False
    except Exception as e:
        return {
            "competition_id": competition_id,
            "metric": None,
            "comment": f"Unexpected error: {str(e)}",
            "success": False,
            "submission_path": str(submission_path)
        }, False


def save_results_to_csv(results: List[Dict], output_file: str, run_folder: Optional[str] = None):
    """Save results to a CSV file with run information."""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['competition_id', 'submission_file', 'metric', 'comment', 'submission_path', 'submission_location_note']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Add metadata as comments in CSV (optional)
        if run_folder:
            csvfile.write(f"# Run folder: {run_folder}\n")
            csvfile.write(f"# Date: {datetime.now().isoformat()}\n")
        
        for result in results:
            writer.writerow({
                'competition_id': result['competition_id'],
                'submission_file': result.get('submission_file', ''),
                'metric': result['metric'],
                'comment': result['comment'],
                'submission_path': result.get('submission_path', ''),
                'submission_location_note': result.get('submission_location_note', '')
            })


def save_run_metadata(run_folder: Path, output_file: str, results: List[Dict], skipped_samples: Dict[str, List[str]]):
    """Save run metadata and results summary."""
    metadata_file = run_folder / "grade_results_metadata.json"
    
    # Group results by competition
    competition_summary = {}
    for r in results:
        comp_id = r['competition_id']
        if comp_id not in competition_summary:
            competition_summary[comp_id] = {
                "competition_id": comp_id,
                "total_submissions": 0,
                "successful_submissions": 0,
                "failed_submissions": 0,
                "skipped_sample_submissions": skipped_samples.get(comp_id, []),
                "submissions": []
            }
        
        comp_summary = competition_summary[comp_id]
        comp_summary["total_submissions"] += 1
        if r.get('success', False):
            comp_summary["successful_submissions"] += 1
        else:
            comp_summary["failed_submissions"] += 1
        
        comp_summary["submissions"].append({
            "file": r.get('submission_file', ''),
            "path": r.get('submission_path', ''),
            "metric": r.get('metric'),
            "success": r.get('success', False),
            "comment": r.get('comment'),
            "location_note": r.get('submission_location_note', '')
        })
    
    metadata = {
        "run_folder": str(run_folder.absolute()),
        "timestamp": datetime.now().isoformat(),
        "total_competitions": len(set(r['competition_id'] for r in results)),
        "total_submissions_graded": len(results),
        "successful_submissions": sum(1 for r in results if r.get('success', False)),
        "failed_submissions": sum(1 for r in results if not r.get('success', False)),
        "skipped_sample_submissions_total": sum(len(samples) for samples in skipped_samples.values()),
        "results_file": output_file,
        "competition_summary": list(competition_summary.values())
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata_file


def print_table(results: List[Dict]):
    """Print results in a formatted table."""
    # Calculate column widths
    max_id_len = max(len(r['competition_id']) for r in results)
    max_file_len = max(len(r.get('submission_file', '')) for r in results)
    max_metric_len = max(len(str(r['metric'])) if r['metric'] is not None else 4 for r in results)
    max_comment_len = max(len(r['comment']) if r['comment'] else 0 for r in results)
    max_note_len = max(len(r.get('submission_location_note', '')) for r in results)
    
    # Set minimum widths
    max_id_len = max(max_id_len, 15)  # "competition_id"
    max_file_len = max(max_file_len, 15)  # "submission_file"
    max_metric_len = max(max_metric_len, 6)  # "metric"
    max_comment_len = max(max_comment_len, 7)  # "comment"
    max_note_len = max(max_note_len, 20)  # "location_note"
    
    # Print header
    print("\n" + "=" * (max_id_len + max_file_len + max_metric_len + max_comment_len + max_note_len + 12))
    print(f"{'Competition ID':<{max_id_len}} | {'Submission File':<{max_file_len}} | {'Metric':<{max_metric_len}} | {'Location Note':<{max_note_len}} | Comment")
    print("-" * (max_id_len + max_file_len + max_metric_len + max_comment_len + max_note_len + 12))
    
    # Print rows
    for result in results:
        metric_str = f"{result['metric']:.5f}" if result['metric'] is not None else "NULL"
        file_name = result.get('submission_file', '')[:max_file_len]
        note = result.get('submission_location_note', '')[:max_note_len]
        print(f"{result['competition_id']:<{max_id_len}} | {file_name:<{max_file_len}} | {metric_str:<{max_metric_len}} | {note:<{max_note_len}} | {result['comment']}")
    
    print("=" * (max_id_len + max_file_len + max_metric_len + max_comment_len + max_note_len + 12))


def discover_competitions_from_folder(run_folder: Path) -> List[str]:
    """
    Discover competition IDs from the folder structure.
    Returns list of competition IDs that have folders in the run folder.
    """
    competitions = []
    
    if not run_folder.exists():
        print(f"Error: Run folder {run_folder} does not exist", file=sys.stderr)
        return []
    
    for item in run_folder.iterdir():
        if item.is_dir():
            competitions.append(item.name)
    
    return sorted(competitions)

def deduplicate_results(results: List[Dict]) -> List[Dict]:
    """
    Deduplicate results for each competition.
    If multiple submissions have the same score, keep only one.
    If they have different scores, keep all.
    """
    # Group by competition_id
    comp_groups = {}
    for result in results:
        comp_id = result['competition_id']
        if comp_id not in comp_groups:
            comp_groups[comp_id] = []
        comp_groups[comp_id].append(result)
    
    # Process each competition
    deduplicated = []
    for comp_id, comp_results in comp_groups.items():
        # Group by metric (score)
        score_groups = {}
        for result in comp_results:
            metric = result.get('metric')
            # Use string representation for None to group nulls together
            score_key = str(metric) if metric is not None else "NULL"
            if score_key not in score_groups:
                score_groups[score_key] = []
            score_groups[score_key].append(result)
        
        # For each score group, keep only the first submission
        for score_key, group_results in score_groups.items():
            if len(group_results) > 1:
                # Multiple submissions with same score
                first_result = group_results[0].copy()
                # Add note about deduplication
                original_comment = first_result.get('comment', '')
                file_names = [r.get('submission_file', '') for r in group_results]
                
                if first_result.get('success', False):
                    first_result['comment'] = f"{original_comment} (Note: {len(group_results)} submissions with identical score {first_result['metric']:.5f} - kept {first_result['submission_file']}, also found: {', '.join(file_names[1:])})"
                else:
                    first_result['comment'] = f"{original_comment} (Note: {len(group_results)} submissions with same result - kept {first_result['submission_file']}, also found: {', '.join(file_names[1:])})"
                
                deduplicated.append(first_result)
            else:
                # Unique score, keep as is
                deduplicated.append(group_results[0])
    
    return deduplicated

def main():
    parser = argparse.ArgumentParser(
        description='Run grade command for multiple competitions - grades all submission files found (excludes sample submissions)',
        epilog='Example: python3 grade_runner.py -r /path/to/2026-03-26_run -o results.csv'
    )
    
    parser.add_argument('-r', '--run-folder', type=str, required=True,
                       help='Run folder containing competition subfolders with submission files')
    parser.add_argument('competitions', nargs='*', help='List of competition IDs (optional, if not provided all will be processed)')
    parser.add_argument('-o', '--output', default='grade_results.csv', 
                       help='Output CSV file (default: grade_results.csv)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed information while running')
    parser.add_argument('-t', '--table', action='store_true',
                       help='Print results in table format in terminal')
    
    args = parser.parse_args()

    # Determine competitions to process
    competitions = []
    run_folder = None
    
    if args.run_folder:
        run_folder = Path(args.run_folder).expanduser().resolve()
        if not run_folder.exists():
            print(f"Error: Run folder {run_folder} does not exist", file=sys.stderr)
            sys.exit(1)
        
        # Auto-discover competitions from folder structure
        competitions = discover_competitions_from_folder(run_folder)
        if args.competitions:
            competitions = [c for c in competitions if c in args.competitions]
        if not competitions:
            print("Error: No competitions provided", file=sys.stderr)
            sys.exit(1)
    
    results = []
    total_submissions_found = 0
    competitions_without_submissions = []
    skipped_samples_by_comp = {}
    
    for i, comp_id in enumerate(competitions, 1):
        if args.verbose:
            print(f"\n[{i}/{len(competitions)}] Processing competition: {comp_id}")
        
        # Find all potential submission files (excluding samples)
        submission_files, skipped_samples = find_all_submission_files(run_folder, comp_id, args.verbose)
        
        # Track skipped samples
        if skipped_samples:
            skipped_samples_by_comp[comp_id] = skipped_samples
        
        if not submission_files:
            # No submission files found
            competitions_without_submissions.append(comp_id)
            
            # Add info about skipped samples if any
            comment = "Agent didn't produce submission"
            if skipped_samples:
                comment = f"Agent didn't produce submission (found {len(skipped_samples)} sample submission file(s) that were skipped)"
            
            result = {
                "competition_id": comp_id,
                "submission_file": "N/A",
                "metric": None,
                "comment": comment,
                "success": False,
                "submission_path": None,
                "submission_location_note": f"No submission file found{' (samples found but skipped)' if skipped_samples else ''}"
            }
            results.append(result)
            
            if not args.verbose:
                if skipped_samples:
                    print(f"✗ {comp_id}: No submission files found (skipped {len(skipped_samples)} sample file(s))")
                else:
                    print(f"✗ {comp_id}: No submission files found")
            continue
        
        total_submissions_found += len(submission_files)
        
        if not args.verbose:
            submission_msg = f"Found {len(submission_files)} submission file(s)"
            if skipped_samples:
                submission_msg += f" (skipped {len(skipped_samples)} sample file(s))"
            print(f"📁 {comp_id}: {submission_msg}")
        
        # Grade each submission file
        for file_idx, (submission_path, location_note) in enumerate(submission_files, 1):
            submission_name = submission_path.name
            
            if args.verbose:
                print(f"\n  [{file_idx}/{len(submission_files)}] Grading: {submission_name}")
                print(f"    Location: {location_note}")
            
            # Check for non-standard submission (for warning)
            is_non_standard = "standard submission" not in location_note.lower()
            
            # Run the grade command
            result, success = run_grade_command(comp_id, submission_path, args.verbose)
            
            # Add additional fields
            result["submission_file"] = submission_name
            result["submission_location_note"] = location_note
            
            # Add warning if non-standard
            if is_non_standard and success:
                result["comment"] = f"Success (non-standard submission - {location_note})"
            elif is_non_standard and not success:
                result["comment"] = f"{result['comment']} (non-standard submission - {location_note})"
            
            results.append(result)
            
            # Print summary for this submission
            if not args.verbose:
                status_icon = "✓" if success else "✗"
                warning_icon = " ⚠" if is_non_standard else ""
                metric_str = f"{result['metric']:.5f}" if result['metric'] else "FAILED"
                print(f"  {status_icon} {submission_name}: {metric_str}{warning_icon}")
    
    # Apply deduplication
    original_count = len(results)
    results = deduplicate_results(results)
    deduplicated = True
    deduplicated_count = len(results)  

    # Save results to CSV
    save_results_to_csv(results, args.output, str(run_folder))
    print(f"\nResults saved to: {args.output}")
    
    # Save run metadata
    metadata_file = save_run_metadata(run_folder, args.output, results, skipped_samples_by_comp)
    print(f"Metadata saved to: {metadata_file}")
    
    # Print summary
    successful_submissions = sum(1 for r in results if r.get('success', False))
    non_standard_submissions = sum(1 for r in results if r.get('submission_location_note') and "standard submission" not in r['submission_location_note'].lower())
    total_skipped_samples = sum(len(samples) for samples in skipped_samples_by_comp.values())
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total competitions processed: {len(competitions)}")
    print(f"Total submission files found (excluding samples): {total_submissions_found}")
    print(f"Total submissions graded: {len(results)}")
    print(f"  - Successful: {successful_submissions}")
    print(f"  - Failed: {len(results) - successful_submissions}")
    if total_skipped_samples > 0:
        print(f"Skipped sample submissions: {total_skipped_samples}")
    print(f"Non-standard submissions: {non_standard_submissions}")
    
    if competitions_without_submissions:
        print(f"\n⚠ Competitions with NO submissions:")
        for comp in competitions_without_submissions:
            if comp in skipped_samples_by_comp:
                print(f"  - {comp} (found {len(skipped_samples_by_comp[comp])} sample file(s) that were skipped)")
            else:
                print(f"  - {comp}")
    
    if total_skipped_samples > 0:
        print(f"\nℹ️ Sample submissions were automatically skipped (not graded)")
        for comp, samples in skipped_samples_by_comp.items():
            print(f"  - {comp}: {', '.join(samples)}")
    
    if non_standard_submissions > 0:
        print(f"\n⚠ WARNING: {non_standard_submissions} non-standard submission(s) were found and graded")
        print(f"   These are submissions that were not in the expected location (root/submission.csv)")
        print(f"   Check the 'submission_location_note' column in the CSV for details")
    
    # Print table if requested
    if args.table:
        print_table(results)
    
    # Return exit code (0 if at least one submission was successful)
    sys.exit(0 if successful_submissions > 0 else 1)


if __name__ == "__main__":
    main()
