import os

base_dir = r"D:\Data_luat_VN\SCRIPT NOW\VieGLUE\data"
required_files = {"train.json", "validation.json"}

def check_task_ready(task_path):
    existing = set(os.listdir(task_path))
    return required_files.issubset(existing)

def main():
    print("📦 Kiểm tra các task trong ViGLUE/data...\n")
    ready = []
    not_ready = []

    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if os.path.isdir(task_path):
            if check_task_ready(task_path):
                print(f"✅ {task}: READY")
                ready.append(task)
            else:
                print(f"❌ {task}: MISSING files → {required_files - set(os.listdir(task_path))}")
                not_ready.append(task)

    print("\n📋 Tổng kết:")
    print(f"✅ Tasks ready: {ready}")
    print(f"❌ Tasks missing files: {not_ready}")

if __name__ == "__main__":
    main()
