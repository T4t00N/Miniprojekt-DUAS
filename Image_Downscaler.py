import os
from PIL import Image

#   Path to the folder containing the images that need dowmsizing:
folder_path = r"C:\Users\anto3\Desktop\DAKI\miniprojekt-DUAS-main\Data\Crowns_martin"

def downscale_image(input_path, output_path, scale_factor=5.5):
    # Open an image file
    with Image.open(input_path) as img:
        # Calculate the new dimensions
        width, height = img.size
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)

        # Resize the image using the LANCZOS resampling algorithm
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save the resized image
        resized_img.save(output_path)

def process_folder(folder_path, scale_factor=5.5):
    # List all files in the folder
    for filename in os.listdir(folder_path):
        # Construct full file path
        input_path = os.path.join(folder_path, filename)
        if os.path.isfile(input_path):
            # Check if the file is an image (by a simple extension check)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                # Create output filename with _DS suffix before the file extension
                base, extension = os.path.splitext(filename)
                output_filename = f"{base}_DS{extension}"
                output_path = os.path.join(folder_path, output_filename)


                downscale_image(input_path, output_path, scale_factor)
                print(f"Processed {filename} to {output_filename}")


process_folder(folder_path)
