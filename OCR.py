import gradio as gr
import pytesseract
import cv2
from PIL import Image
import numpy as np
import os
from google.colab import files

# Step 1: Upload your images from your local folder (C:\GIRAF\Images)
uploaded = files.upload()  # Manually select images like Sleep.jpg, Eat.jpg, Dance.jpg from your local folder

# Step 2: Create a folder to store the uploaded images
os.makedirs('image_database', exist_ok=True)

# Move uploaded images into the folder
for filename in uploaded.keys():
    os.rename(filename, os.path.join('image_database', filename))

# Step 3: Create an image database with file paths pointing to the uploaded images
image_database = {
    "Eat": "/content/image_database/Eat.jpg",
    "Sleep": "/content/image_database/Sleep.jpg",
    "Dance": "/content/image_database/Dance.jpg"
}

# Function to extract text and find multiple matching images
def extract_text_from_image(image):
    # Convert the image to a format suitable for OpenCV and Tesseract
    image = np.array(image)  # Convert PIL image to NumPy array
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run Tesseract OCR on the image
    extracted_text = pytesseract.image_to_string(image_rgb).strip()

    # DEBUG: Print extracted text
    print(f"Extracted Text: {extracted_text}")

    # Convert extracted text to lowercase for comparison
    extracted_text_lower = extracted_text.lower()

    # List to store matched images
    matched_images = []

    # Search for multiple keywords in the extracted text
    for keyword in image_database:
        if keyword.lower() in extracted_text_lower:  # Check if the keyword is in the extracted text
            matched_image_path = image_database[keyword]

            # Check if the file exists at the path
            if os.path.exists(matched_image_path):
                # DEBUG: Print found image
                print(f"Found image for keyword: '{keyword}' at '{matched_image_path}'")
                matched_images.append(matched_image_path)  # Append matched image path to the list

    # If no images are matched, return a message
    if not matched_images:
        return f"Extracted Text: {extracted_text}\nNo matching images found in the database.", []

    # Return the extracted text and the list of matched image paths
    return f"Extracted Text: {extracted_text}", matched_images

# Create a Gradio Blocks interface
with gr.Blocks() as iface:
    gr.Markdown("# Image to Text with Multiple Matches")
    gr.Markdown("Upload an image to extract text and find multiple matching images from the database.")

    # Input component
    image_input = gr.Image(type="pil", label="Upload Image")

    # Button to trigger extraction
    submit_btn = gr.Button("Extract Text and Find Images")

    # Output components
    text_output = gr.Textbox(label="Extracted Text", interactive=False)

    # Use Gallery for images (display vertically)
    image_output = gr.Gallery(label="Matched Images", show_label=False)

    # Set up the interaction
    submit_btn.click(extract_text_from_image, inputs=image_input, outputs=[text_output, image_output])

# Launch the interface
iface.launch()
