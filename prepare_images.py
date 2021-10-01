import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

input_train_folder = "./data/train/row_data/train"
output_train_folder = "./data/train/prepared_data"
input_test_folder = "./data/test/row_data/test"
output_test_folder = "./data/test/prepared_data"


def resize_image(input_folder, output_folder, resize):
    base_name = os.path.basename(input_folder)
    output_path = os.path.join(output_folder, base_name)
    img = Image.open(input_folder)
    img = img.resize(
        (resize[1], resize[0]), resample=Image.BILINEAR
    )
    img.save(output_path)


def resize_train_images():
    images = glob.glob(os.path.join(input_train_folder, "*.jpg"))
    Parallel(n_jobs=8)(
        delayed(resize_image)(
            image,
            output_train_folder,
            (512, 512)
        ) for image in tqdm(images)
    )


def resize_test_images():
    images = glob.glob(os.path.join(input_test_folder, "*.jpg"))
    Parallel(n_jobs=8)(
        delayed(resize_image)(
            image,
            output_test_folder,
            (512, 512)
        ) for image in tqdm(images)
    )


if __name__ == "__main__":
    resize_train_images()
    resize_test_images()
