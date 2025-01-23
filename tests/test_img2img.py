import requests
import io
import json
from PIL import Image

# SCHEDULER OPTIONS:
# Euler
# Euler Karras
# Euler Ancestral
# Heun
# Heun Karras
# DPM++ 2M
# DPM++ 2M Karras
# DPM++ 2M SDE
# DPM++ 2M SDE Karras
# K DPM 2
# K DPM 2 Karras
# K DPM 2 Ancestral
# UniPC 2M
# UniPC 2M Karras
# LMS
# LMS Karras

def test_image_to_image():
    url = "http://localhost:8000/image-to-image"
    
    # Open source image
    try:
        source_image_path = "source_image.png"
        source_image = Image.open(source_image_path)
    except Exception as e:
        print(f"Error loading source image: {str(e)}")
        return

    # Convert image to bytes
    image_bytes = io.BytesIO()
    source_image.save(image_bytes, format=source_image.format or 'PNG')
    image_bytes.seek(0)

    # Prepare the multipart form data
    files = {
        'image': ('source_image.png', image_bytes, 'image/png')
    }
    
    # Example of supported payload parameters
    payload = {
        "model_name": "model name only (no extension)",
        "prompt": "",
        "negative_prompt": "",
        "strength": 0.7,
        "steps": 30,
        "guidance": 6,
        "seed": None,
        "scheduler_name": "DPM++ 2M SDE Karras",
        "loras": [
            {"name": "model name only (no extension)", "weight": 1.2},
            # { "name": "add more", "weight": 1.5 },
        ]
    }
    
    try:
        print("Sending request...")
        request_json = json.dumps(payload)
        
        response = requests.post(
            url,
            files=files,
            data={'request': request_json}
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Saving image...")
            output_image = Image.open(io.BytesIO(response.content))
            output_image.save("test_img2img_output.png")
            print("Image saved as 'test_img2img_output.png'")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_image_to_image()