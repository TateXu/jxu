import os
import time

def timer(func):
    def wrapper(*args, **kargs):
        start_time = time.time()
        f = func(*args, **kargs)
        exec_time = time.time() - start_time
        print(exec_time)
        return f
    return wrapper

def create_folder(folder_path):


    if os.path.exists(folder_path):
        # print("Traget folder is already created! Path: \n" + folder_path)
        return 0
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

