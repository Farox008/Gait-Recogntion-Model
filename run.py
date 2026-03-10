import os
import sys
import subprocess

def main():
    while True:
        print("\n" + "="*50)
        print(" Gait Model Interactive Runner")
        print("="*50)
        print("1. Train (Enroll) only")
        print("2. Run (Test) only")
        print("3. Both (Train then Test)")
        print("4. Custom Directory Test")
        print("5. Single Video Test")
        print("6. Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break
            
        if choice == '1':
            print("\nStarting Training (Enrollment)...")
            subprocess.run([sys.executable, "enroll_and_test.py", "--mode", "train"])
        elif choice == '2':
            print("\nStarting Run (Testing)...")
            subprocess.run([sys.executable, "enroll_and_test.py", "--mode", "test"])
        elif choice == '3':
            print("\nStarting Both (Train then Test)...")
            subprocess.run([sys.executable, "enroll_and_test.py", "--mode", "both"])
        elif choice == '4':
            dir_path = input("Enter the path to the data directory: ").strip()
            if os.path.isdir(dir_path):
                print(f"\nStarting Custom Directory Test on: {dir_path}")
                subprocess.run([sys.executable, "enroll_and_test.py", "--mode", "test", "--vods-dir", dir_path])
            else:
                print(f"Error: {dir_path} is not a valid directory.")
        elif choice == '5':
            file_path = input("Enter the path to the video file: ").strip()
            if os.path.isfile(file_path):
                print(f"\nStarting Single Video Test on: {file_path}")
                subprocess.run([sys.executable, "test_single_clip.py", file_path])
            else:
                print(f"Error: {file_path} is not a valid file.")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
