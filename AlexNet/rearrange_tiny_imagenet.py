import os, glob

IMAGENET_DIR = r'D:\work\tiny-imagenet-200'
for root, dirs, files in os.walk(IMAGENET_DIR):
    if 'train' in root and 'images' in root:
        class_dir, _ = os.path.split(root)
        print(f'moving for : {class_dir}')

        for txtfile in glob.glob(os.path.join(class_dir, "*.txt")):
            os.remove(txtfile)

        for img_file in files:
            original_path = os.path.join(root, img_file)
            new_path = os.path.join(class_dir, img_file)
            os.rename(original_path, new_path)
        os.rmdir(root)


# for root, dirs, files in os.walk(IMAGENET_DIR):
#     if 'train' in root and 'images' in root:
#         # print(root)
#         class_dir, _ = os.path.split(root)
#         # print(class_dir, _)
#         for txtfile in glob.glob(os.path.join(class_dir, "*.txt")):
#             # print(txtfile)
#
#         for img_file in files:
#             original_path = os.path.join(root, img_file)
#             # print(original_path)
#             new_path = os.path.join(class_dir, img_file)
#             # print(new_path)
