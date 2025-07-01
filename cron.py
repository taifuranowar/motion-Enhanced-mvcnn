#!/usr/bin/env python3
import subprocess
import time
import sys

def run_command(command):
    """
    Run a command and stream its output to the terminal
    """
    print("\n" + "="*80)
    print(f"RUNNING COMMAND: {command}")
    print("="*80 + "\n")
    
    # Wait a moment to ensure the separation is visible
    time.sleep(1)
    
    try:
        # Run the command and stream output to the terminal
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output line by line to the terminal
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        
        # Wait for the process to complete and get the return code
        return_code = process.wait()
        
        if return_code != 0:
            print(f"\nCommand failed with return code {return_code}")
            return False
        
        print("\nCommand completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError executing command: {e}")
        return False

def main():
    # List of commands to run
    commands = [
        'python hello_verbose.py',
        'python mvcnn_training.py --dataset-path "mvcnn_dataset" --output-dir "mvcnn_results" --backbone resnet18  --epochs 30 --use-pretrained --max-views 12 --batch-size 8 --elevations "0,30" --num-classes 10',
        'python mvcnn_training.py --dataset-path "mvcnn_dataset" --output-dir "mvcnn_results" --backbone resnet18  --epochs 30 --use-pretrained --max-views 24 --batch-size 4 --elevations "0,30" --num-classes 10',
        'python mvcnn_training.py --dataset-path "mvcnn_dataset" --output-dir "mvcnn_results" --backbone resnet18  --epochs 30 --use-pretrained --max-views 48 --batch-size 4 --elevations "0,30" --num-classes 10',
        'python motion_mvcnn_training.py --dataset-path "mvcnn_dataset" --num-classes 10 --motion-mode learnable --max-views 12 --batch-size 8 --elevations "0,30"',
        'python motion_mvcnn_training.py --dataset-path "mvcnn_dataset" --num-classes 10 --motion-mode learnable --max-views 24 --batch-size 3 --elevations "0,30"',
        'python motion_mvcnn_training.py --dataset-path "mvcnn_dataset" --num-classes 10 --motion-mode learnable --max-views 48 --batch-size 3 --elevations "0,30"'
    ]
    
    # Run each command one by one
    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] Running next command...")
        success = run_command(cmd)
        
        if not success:
            print("\nCommand failed. Automatically continuing to next command...")
            # Optional: Add a short pause so you can see the failure message
            time.sleep(2)
        
        # Wait a moment before starting the next command
        if i < len(commands):
            print("\nWaiting before starting next command...")
            time.sleep(3)
    
    print("\nAll commands completed!")

if __name__ == "__main__":
    main()