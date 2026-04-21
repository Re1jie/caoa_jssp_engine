import pandas as pd
import argparse
import os

def detect_machine_usage(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
        
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Check for expected columns
    expected_cols = ['job_id', 'machine_id', 'A_lj', 'p_lj']
    for col in expected_cols:
        if col not in df.columns:
            print(f"Error: Column {col} not found in the CSV.")
            return

    # Calculate end time of the operation
    df['End_Time'] = df['A_lj'] + df['p_lj']

    # 1. Detect re-visitation by the same job (ship)
    revisits = []
    for (job_id, machine_id), group in df.groupby(['job_id', 'machine_id']):
        if len(group) > 1:
            revisits.append({
                'job_id': job_id,
                'machine_id': machine_id,
                'visit_count': len(group)
            })
            
    if revisits:
        revisits_df = pd.DataFrame(revisits)
        print(f"\n--- Detected {len(revisits)} Cases of Ship Revisiting the Same Port ---")
        print(revisits_df.head(20).to_string(index=False))
        if len(revisits) > 20:
             print(f"... and {len(revisits)-20} more revisits.")
    else:
        print("\n--- No ships revisit the same port. ---")

    # 2. Detect overlaps (Congestion) on the same machine across different jobs
    overlaps = []
    
    for machine_id, group in df.groupby('machine_id'):
        if len(group) <= 1:
            continue
        
        # Sort operations on this machine by start time
        sorted_group = group.sort_values(by='A_lj')
        
        records = sorted_group.to_dict('records')
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                op1 = records[i]
                op2 = records[j]
                
                # Check for overlap: does op2 start before op1 ends?
                if op2['A_lj'] < op1['End_Time']:
                    # We might only care about collisions between different ships
                    if op1['job_id'] != op2['job_id']:
                        overlaps.append({
                            'machine_id': machine_id,
                            'job_1': op1['job_id'],
                            'op_seq_1': op1.get('op_seq', ''),
                            'start_1': op1['A_lj'],
                            'end_1': op1['End_Time'],
                            'job_2': op2['job_id'],
                            'op_seq_2': op2.get('op_seq', ''),
                            'start_2': op2['A_lj'],
                            'end_2': op2['End_Time']
                        })
                else:
                    # Because data is sorted by A_lj, if op2 starts after op1 ends,
                    # all subsequent operations will also start after op1 ends.
                    break

    if overlaps:
        overlap_df = pd.DataFrame(overlaps)
        # Sort by machine and then start time
        overlap_df = overlap_df.sort_values(by=['machine_id', 'start_1'])
        print(f"\n--- Detected {len(overlaps)} Overlapping Usages (Congestion) on Same Machine ---")
        print(overlap_df.head(20).to_string(index=False))
        if len(overlaps) > 20:
            print(f"... and {len(overlaps)-20} more overlaps. Output saved to CSV.")
            
        # Save output to same directory as input
        out_dir = os.path.dirname(file_path)
        out_file = os.path.join(out_dir, "machine_overlaps.csv")
        overlap_df.to_csv(out_file, index=False)
        print(f"\nFull overlap report saved to {out_file}")
    else:
        print("\n--- No overlapping time usages (congestion) detected. ---")

    # 3. Overall Machine Load
    print("\n--- Top 10 Most Heavily Used Machines ---")
    usage_counts = df['machine_id'].value_counts()
    print("Machine ID | Visit Count")
    for m_id, count in usage_counts.head(10).items():
        print(f"{m_id:<10} | {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect same machine/port usage and time overlaps.")
    parser.add_argument("--file", type=str, default="data/processed/jssp_data_sliced.csv", 
                        help="Path to the sliced CSV file")
    args = parser.parse_args()
    
    detect_machine_usage(args.file)
