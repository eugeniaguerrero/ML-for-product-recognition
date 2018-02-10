from src.DATA_PREPARATION.folder_manipulation import *

directory_ = "F:\product_image_dataset_unpacked"
folders = get_folders(directory_)

for folder in folders:
    file = os.listdir(os.path.join(directory_,folder))
    print(file)
    print(os.path.join(directory_,folder,file[0]))
    os.rename(os.path.join(directory_,folder,file[0]),os.path.join(directory_,"_" + file[0]))

