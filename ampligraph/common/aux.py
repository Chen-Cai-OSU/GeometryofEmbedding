import os
def make_dir(dir):
    # has side effect
    if not os.path.exists(dir):
        os.makedirs(dir)
