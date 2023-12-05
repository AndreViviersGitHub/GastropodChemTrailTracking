import os
import random
import shutil



if __name__ == '__main__':
    # Set the source directory containing your images
    source_directory = "E:\\Snail Images\\RandomBS\\Cluster_5"

    # Set the target directories for the 80% and 20% groups
    group_80_directory = "E:\\Snail Images\\Training\\Training\\pipeEdge"
    group_20_directory = "E:\\Snail Images\\Validation\\Validation\\pipeEdge"

    # Ensure the target directories exist or create them
    os.makedirs(group_80_directory, exist_ok=True)
    os.makedirs(group_20_directory, exist_ok=True)

    # Get a list of all image files in the source directory
    image_files = [f for f in os.listdir(source_directory) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Calculate the number of images for each group
    total_images = len(image_files)
    num_images_80_percent = int(0.8 * total_images)
    num_images_20_percent = total_images - num_images_80_percent

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Split the images into the two groups
    group_80 = image_files[:num_images_80_percent]
    group_20 = image_files[num_images_80_percent:]

    # Copy images to the 80% and 20% directories
    copy_count = 0

    # Copy images to the 80% directory
    for image in group_80:
        source_path = os.path.join(source_directory, image)
        target_path = os.path.join(group_80_directory, image)
        #shutil.copy(source_path, target_path)
        copy_count += 1
        #print(f"Copying image {copy_count} to {group_80_directory}...")

    # Reset the copy count
    copy_count = 0

    # Copy images to the 20% directory
    for image in group_20:
        source_path = os.path.join(source_directory, image)
        target_path = os.path.join(group_20_directory, image)
        shutil.copy(source_path, target_path)
        copy_count += 1
        print(f"Copying image {copy_count} to {group_20_directory}...")

    print(f"{num_images_80_percent} images copied to {group_80_directory}")
    print(f"{num_images_20_percent} images copied to {group_20_directory}")