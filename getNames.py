import os
import os.path
# get the names or dataset
path = './vis_feature/test_data/images'
txt = './vis_feature/test_data/test.txt'
file = open(txt,'w')

for dirpath, dirnames,filenames in os.walk(path):
    for names in filenames:
        name = names.split('.')[0]
        if names.split('.')[-1] == 'jpg':
            file.write(name + '\n')
        # print(name)
    file.close()
print('done!')    
