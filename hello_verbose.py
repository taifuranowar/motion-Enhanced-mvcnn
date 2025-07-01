import time
import sys
from datetime import datetime

def print_banner():
    print("\n" + "="*60)
    print(" " * 15 + "\033[92mHELLO WORLD - JOB ROTATION TEST\033[0m")
    print("="*60 + "\n")

def main():
    print_banner()
    for i in range(5):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] \033[94mHello World!\033[0m Loop {i+1}/5")
        sys.stdout.flush()
        time.sleep(1)
    print("\n\033[93mHello World script finished successfully!\033[0m\n")

if __name__ == "__main__":
    main()