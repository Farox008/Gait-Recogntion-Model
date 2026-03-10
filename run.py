import os
import sys
import subprocess

def get_camera_id():
    cam_id = input("\nEnter Camera ID (or press Enter to use default global gallery): ").strip()
    return cam_id if cam_id else None

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
            cam_id = get_camera_id()
            print("\nStarting Training (Enrollment)...")
            cmd = [sys.executable, "enroll_and_test.py", "--mode", "train"]
            if cam_id: cmd.extend(["--camera-id", cam_id])
            subprocess.run(cmd)
        elif choice == '2':
            cam_id = get_camera_id()
            print("\nStarting Run (Testing)...")
            cmd = [sys.executable, "enroll_and_test.py", "--mode", "test"]
            if cam_id: cmd.extend(["--camera-id", cam_id])
            subprocess.run(cmd)
        elif choice == '3':
            cam_id = get_camera_id()
            print("\nStarting Both (Train then Test)...")
            cmd = [sys.executable, "enroll_and_test.py", "--mode", "both"]
            if cam_id: cmd.extend(["--camera-id", cam_id])
            subprocess.run(cmd)
        elif choice == '4':
            dir_path = input("Enter the path to the data directory: ").strip()
            if os.path.isdir(dir_path):
                cam_id = get_camera_id()
                print(f"\nStarting Custom Directory Test on: {dir_path}")
                cmd = [sys.executable, "enroll_and_test.py", "--mode", "test", "--vods-dir", dir_path]
                if cam_id: cmd.extend(["--camera-id", cam_id])
                subprocess.run(cmd)
            else:
                print(f"Error: {dir_path} is not a valid directory.")
        elif choice == '5':
            file_path = input("Enter the path to the video file: ").strip()
            if os.path.isfile(file_path):
                cam_id = get_camera_id()
                print(f"\nStarting Single Video Test on: {file_path}")
                cmd = [sys.executable, "test_single_clip.py", file_path]
                if cam_id: cmd.extend(["--camera-id", cam_id])
                subprocess.run(cmd)
            else:
                print(f"Error: {file_path} is not a valid file.")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()
