from matplotlib import pyplot as plt
import os
from PIL import Image
import warnings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


# Suppress all warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")
# # # show number or rows
# print(ds.num_rows)

# # # get an image from the dataset
# flower = ds["train"][78]["image"]
# Display the image using matplotlib
# plt.imshow(flower)
# plt.axis("off")
# plt.show()


def show_image_from_uri(uri):
    # Open the image using PIL
    img = Image.open(uri)

    # Display the image using matplotlib
    plt.imshow(img)
    plt.axis("off")  # Turn off axis labels
    plt.show()


# ==== Save all images (500) do directory ====
base_dir = Path(__file__).resolve().parent
dataset_folder = base_dir / "dataset" / "flowers-102-categories"
os.makedirs(dataset_folder, exist_ok=True)


# Function to save images
def save_images(dataset, dataset_folder, num_images=500):
    for i in range(num_images):
        print(f"Saving image {i+1} of {num_images}")
        # Get the image data
        image = dataset["train"][i]["image"]

        # Save the image
        image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))

    print(f"Saved the first 500 images to {dataset_folder}")

# save_images(ds, dataset_folder, num_images=500)  # Uncomment to save images