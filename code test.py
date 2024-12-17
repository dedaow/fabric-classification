import os

material_names = ['glass', 'white_foam', 'blue_foam', 'woven_fabric',
                  'gray_sponge', 'ribbed_fabric', 'yarn_screen', 'glossy_wood', 'rough_wood',
                  'Popline', 'twill', ' Polyester and Spandex', ' cotton Oxford fabric',
                  'olyester mountain climbing ', 'Dobby', 'Nylon striped fabric', 'Stretch twill',
                  'Crimp cloth jacquard fabric', 'plush fabric', 'Combed Cotton Modal ', 'Tulle Mesh']

print(material_names[16])
root_path = 'data/train'
for file_name in os.listdir(root_path):
    if material_names[16] in file_name:
        print(f'{file_name} is exist.')