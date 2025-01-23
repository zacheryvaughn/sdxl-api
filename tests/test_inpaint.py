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

def test_inpaint():
    url = "http://localhost:8000/inpaint"
    
    # Open source image and mask
    try:
        source_image_path = "source_image.png"
        mask_image_path = "mask_image.png"
        source_image = Image.open(source_image_path)
        mask_image = Image.open(mask_image_path)
    except Exception as e:
        print(f"Error loading images: {str(e)}")
        return

    # Convert images to bytes
    source_bytes = io.BytesIO()
    mask_bytes = io.BytesIO()
    source_image.save(source_bytes, format=source_image.format or 'PNG')
    mask_image.save(mask_bytes, format=mask_image.format or 'PNG')
    source_bytes.seek(0)
    mask_bytes.seek(0)

    # Prepare the multipart form data
    files = {
        'image': ('source_image.png', source_bytes, 'image/png'),
        'mask': ('mask_image.png', mask_bytes, 'image/png')
    }
    
    # Example of supported payload parameters
    payload = {
        "model_name": "model name only (no extension)",
        "prompt": "",
        "negative_prompt": "",
        "steps": 30,
        "guidance": 6,
        "seed": None,
        "scheduler_name": "DPM++ 2M SDE Karras",
        "loras": [
            {"name": "model name only (no extension)", "weight": 1.2},
            # { "name": "add more", "weight": 1.5 },
        ]
    }

    
    data = {
        'request': json.dumps(payload)
    }
    
    try:
        print("Sending request...")
        
        response = requests.post(
            url,
            files=files,
            data=data
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Saving image...")
            output_image = Image.open(io.BytesIO(response.content))
            output_image.save("test_inpaint_output.png")
            print("Image saved as 'test_inpaint_output.png'")
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"Request error occurred: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # Clean up BytesIO objects
        source_bytes.close()
        mask_bytes.close()

if __name__ == "__main__":
    test_inpaint()