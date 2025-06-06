import os
import tarfile
from pathlib import Path
import shutil

base_dir = r"D:\Data_luat_VN\SCRIPT NOW\VieGLUE\data"

def safe_extract_tar(tar_path, output_dir):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

def normalize_filename(filename):
    # Đổi tên về đúng chuẩn Hugging Face
    if "train" in filename.lower():
        return "train.json"
    elif "dev" in filename.lower() or "val" in filename.lower():
        return "validation.json"
    elif "test" in filename.lower():
        return "test.json"
    else:
        return filename

def process_folder(task_folder):
    files = list(Path(task_folder).glob("*.tar.gz"))
    for f in files:
        extract_temp = os.path.join(task_folder, "temp_extract")
        os.makedirs(extract_temp, exist_ok=True)
        print(f"🔍 Extracting {f.name}...")
        safe_extract_tar(f, extract_temp)

        for extracted in os.listdir(extract_temp):
            src = os.path.join(extract_temp, extracted)
            dst = os.path.join(task_folder, normalize_filename(extracted))
            shutil.move(src, dst)
            print(f"✅ Moved: {extracted} → {os.path.basename(dst)}")

        shutil.rmtree(extract_temp)

if __name__ == "__main__":
    for task in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task)
        if os.path.isdir(task_path):
            print(f"\n📂 Processing task: {task}")
            process_folder(task_path)

    print("\n🎉 All data extracted and normalized.")
