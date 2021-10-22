import os
import glob
from PIL import Image, ImageOps

# input_dirname = os.path.join('C:/Users/Beppy/model_tutorial/GAN/dataset/labeled/valids/others_pre/*')
# output_dirname = os.path.join('C:/Users/Beppy/model_tutorial/GAN/dataset/labeled/valids/others')

input_dirname = os.path.join('C:/Users/Beppy/model_tutorial/GAN/downloads/*/*')
output_dirname = os.path.join('C:/Users/Beppy/model_tutorial/GAN/dataset/reshaped')

files = glob.glob(input_dirname)
reshaped_size = (256, 256)
print(input_dirname)
test = os.path.exists(input_dirname)
print(test)

for i, file in enumerate(files):
    index = i + 1
    try:
        image = Image.open(file)
    except IOError:
        pass
    reshaped = ImageOps.fit(image, reshaped_size, Image.NEAREST)
    converted = reshaped.convert('RGB')
    converted.save(os.path.join(output_dirname, f'{index}.jpg'))
    print(f'{index}: {file} was saved.')