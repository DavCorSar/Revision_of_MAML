import os
import shutil
from config import DATA_PATH

path = DATA_PATH + '/Omniglot/'
pat = ['old_images_background', 'old_images_evaluation']

for filer in pat:
    for alphabet in os.listdir(path+f'{filer}'):
        
        
        for classe in os.listdir(path+'{}/{}'.format(filer, alphabet)):
            
        
            for image in os.listdir(path+'{}/{}/{}/'.format(filer, alphabet, classe)):
                # print(path+'{}/{}/{}/{}'.format(filer, alphabet, classe, image))
                shutil.move(path+'{}/{}/{}/{}'.format(filer, alphabet, classe, image), path+'{}/{}'.format(filer[4:], image))
        
