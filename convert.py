import os 
import glob
from PIL import Image


path = os.path.join(os.getcwd(), 'train', 'Yosemite')

trainA_path = os.path.join(path, 'testA') + '/*.jpg'

images = glob.glob(trainA_path)
images = sorted(images)
print(len(images))

for image in images:
    i = Image.open(image).convert('RGB').resize((128, 128))
    i.save(image, format='JPEG', subsampling=0, quality=100)