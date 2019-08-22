import os

if __name__ == '__main__':

    data_dir = 'dstc2_data'
    file_name = 'dstc2_traindev.tar.gz'
    url = 'http://camdial.org/~mh521/dstc/downloads/%s' % file_name
    file = '%s/%s' % (data_dir, file_name)

    if not os.path.isdir(data_dir):
        os.system('wget -N -q -P %s %s' % (data_dir, url))
        os.system('tar xvzf %s -C %s'%(file,data_dir) )
        os.remove(file)