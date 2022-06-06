import os
import shutil
from config import DATA_PATH

path = DATA_PATH + '/miniImageNet/images'

for filer in os.listdir(path):
    
    for image in os.listdir(path+'/{}'.format(filer)):
        
        shutil.move(path+'/{}/{}'.format(filer, image), path+'/{}'.format(image))
