import os


def create_folder(folder_path):


    if os.path.exists(folder_path):
        return print("Traget folder is already created! Path: \n" + folder_path)
    else:
        try:
            os.mkdir(folder_path)
        except:
            path = os.path.normpath(folder_path)
            path_seg = path.split(os.sep)
            if len(path_seg) == 0:
                raise ValueError("Empty path of target folder!")

            for subfolder_path in path_seg:
                if not os.path.exists(subfolder_path):
                    os.mkdir(subfolder_path)
                os.chdir(subfolder_path)
            os.chdir(os.path.dirname(os.path.abspath(os.getcwd())))

        return print("Target folder is successfully created!")

