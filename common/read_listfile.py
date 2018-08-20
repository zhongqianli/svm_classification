
def read_filename_list(filename):
    file = open(filename)
    filename_list = []
    while True:
        filename = file.readline()
        if not filename:
            break
        filename_list.append(filename.strip('\n'))
    return filename_list

def read_listfile(listfiles):
    filename_list = []
    for filename in listfiles:
        filename_list.extend(read_filename_list(filename))

    return filename_list

if __name__ == '__main__':
    listfiles = ['../config/carpet/train/class1.lst', '../config/carpet/train/class2.lst']
    filename_list = read_listfile(listfiles)
    for filename in filename_list:
        print(filename)
