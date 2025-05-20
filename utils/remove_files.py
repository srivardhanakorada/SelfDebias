import os
import re

def remove_original_files(folder_path):
    pattern = re.compile(r'^original_\d{4}\.png$')
    for filename in os.listdir(folder_path):
        if pattern.match(filename):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

remove_original_files('our_outputs/original')