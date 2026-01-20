import pandas as pd
import re
from datetime import datetime

def analyze_log_timing(log_file_path):
    """
    Extract datetimes from log file and compute time differences between 
    'Method show_now called.' entries.
    
    Args:
        log_file_path: Path to the log file
    """
    # Read the log file
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Pattern to match datetime and the specific log message
    pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\|.*Method show_now called\.'
    
    # Extract datetimes and line numbers for "Method show_now called." entries
    datetimes = []
    line_numbers = []
    for i, line in enumerate(lines):
        match = re.match(pattern, line)
        if match:
            dt_str = match.group(1)
            dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S.%f')
            datetimes.append(dt)
            line_numbers.append(i)
    
    print(f"Found {len(datetimes)} 'Method show_now called.' entries\n")
    
    if len(datetimes) < 2:
        print("Need at least 2 entries to compute time differences.")
        return
    
    # Convert to pandas Series
    dt_series = pd.Series(datetimes)
    
    # Compute time differences (in seconds)
    time_diffs = dt_series.diff().dt.total_seconds()
    
    # Remove the first NaN value
    time_diffs_clean = time_diffs.dropna()
    
    print("Time differences between consecutive 'Method show_now called.' entries:")
    print(time_diffs_clean.describe())
    
    # Find the longest gap
    max_gap_idx = time_diffs.idxmax()  # Index in original series (includes NaN at 0)
    max_gap = time_diffs[max_gap_idx]
    
    print(f"\n{'='*80}")
    print(f"LONGEST GAP: {max_gap:.3f} seconds")
    print(f"Between entry #{max_gap_idx-1} and entry #{max_gap_idx}")
    print(f"{'='*80}\n")
    
    # Get the line numbers for the entries before and after the gap
    line_before = line_numbers[max_gap_idx - 1]
    line_after = line_numbers[max_gap_idx]
    
    # Calculate range to show (5 lines before the first entry, 5 lines after the second)
    start_line = max(0, line_before - 5)
    end_line = min(len(lines), line_after + 6)  # +6 to include 5 lines after
    
    print(f"Context around the longest gap (lines {start_line+1} to {end_line}):")
    print(f"Time before gap: {datetimes[max_gap_idx-1]}")
    print(f"Time after gap:  {datetimes[max_gap_idx]}")
    print(f"\n{'-'*80}")
    
    for i in range(start_line, end_line):
        # Highlight the two entries that bracket the gap
        if i == line_before:
            print(f">>> LINE {i+1} (BEFORE GAP) <<<")
        elif i == line_after:
            print(f">>> LINE {i+1} (AFTER GAP) <<<")
        
        print(f"{i+1:5d}: {lines[i]}", end='')
    
    print(f"{'-'*80}\n")
    
    return time_diffs_clean

if __name__ == "__main__":
    # Example usage with your log data
    log_file = "sample_log.txt"  # Replace with your actual log file path
    
    # Analyze the log file
    time_diffs = analyze_log_timing("sample_log.txt")
