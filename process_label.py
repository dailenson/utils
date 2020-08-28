import os
import shutil

path = "/home/datasets/PACS"
def process_data(source_dir, dist_dir):
    for file_name in os.listdir(source_dir):
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(dist_dir, file_name))
        if "train" in file_name:
            file_path = os.path.join(source_dir, file_name)
            with open(file_path,'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                    _, label = [i for i in lines.split()] #label的存放形式是数据名称+对应label，中间用空格隔开
                    if int(label) < 4:
                        temp_name = "meta_test_" + file_name
                        with open(os.path.join(dist_dir, temp_name), "a") as f:
                            f.write(lines)
            print(file_name)

if __name__ == "__main__":
    source_dir = os.path.join(path, "pacs_label_backup")
    dist_name = "pacs_label_pre"
    dist_dir = os.path.join(path, dist_name)
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    process_data(source_dir, dist_dir)
