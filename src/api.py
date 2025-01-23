"""
SDXL API Server

This module provides a FastAPI server for Stable Diffusion XL operations.
It exposes endpoints for text-to-image, image-to-image, and inpainting operations,
with a focus on memory efficiency and zero file storage.

Key Features:
- Memory-efficient operation with no file storage
- Support for multiple SDXL operations
- Streaming response for generated images
- Comprehensive error handling
- Health check endpoint

API Endpoints:
- POST /text-to-image: Generate image from text prompt
- POST /image-to-image: Transform existing image using prompt
- POST /inpaint: Modify specific areas of image using mask
- GET /health: Check API health status

Example Usage:
    ```bash
    # Text to Image
    curl -X POST "http://localhost:8000/text-to-image" \
         -H "Content-Type: application/json" \
         -d '{"model_name": "model1", "prompt": "a photo of a cat"}'
    
    # Image to Image
    curl -X POST "http://localhost:8000/image-to-image" \
         -F "image=@source.png" \
         -F 'request={"model_name": "model1", "prompt": "a photo of a dog"}'
    
    # Inpainting
    curl -X POST "http://localhost:8000/inpaint" \
         -F "image=@source.png" \
         -F "mask=@mask.png" \
         -F 'request={"model_name": "model1", "prompt": "a photo of a bird"}'
    ```
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import json
from io import BytesIO
from PIL import Image
import gc
from inference import SDXLConfig, LoRAConfig, SDXLInference, OperationType

# Initialize FastAPI app with metadata
app = FastAPI(
    title="SDXL API",
    description="API for Stable Diffusion XL image generation and manipulation",
    version="1.0.0"
)

# Initialize inference engine
sdxl = SDXLInference()

class LoRARequest(BaseModel):
    """
    Request model for LoRA configuration.
    
    Attributes:
        name (str): Name of the LoRA model to use
        weight (float): Weight to apply to the LoRA model (default: 1.0)
    """
    name: str
    weight: float = 1.0

class BaseSDXLRequest(BaseModel):
    """
    Base request model for all SDXL operations.
    
    Attributes:
        model_name (str): Name of the main model to use
        prompt (str): Text prompt describing desired output
        negative_prompt (str): Text prompt describing elements to avoid
        steps (int): Number of inference steps
        guidance (float): Classifier-free guidance scale
        seed (int): Random seed for reproducibility
        scheduler_name (str): Name of the scheduler to use
        loras (List[LoRARequest]): List of LoRA models to apply
    """
    model_name: str
    prompt: str
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = 30
    guidance: Optional[float] = 7.0
    seed: Optional[int] = None
    scheduler_name: Optional[str] = "DPM++ 2M"
    loras: Optional[List[LoRARequest]] = []

class TextToImageRequest(BaseSDXLRequest):
    """
    Request model for text-to-image operation.
    
    Additional Attributes:
        width (int): Output image width in pixels
        height (int): Output image height in pixels
    """
    width: Optional[int] = 768
    height: Optional[int] = 1024

class ImageToImageRequest(BaseSDXLRequest):
    """
    Request model for image-to-image operation.
    
    Additional Attributes:
        strength (float): Transformation strength (0.0 to 1.0)
    """
    strength: Optional[float] = 0.7

class InpaintRequest(BaseSDXLRequest):
    """Request model for inpainting operation."""
    pass

def image_to_bytes(image: Image.Image) -> bytes:
    """
    Converts PIL Image to bytes for streaming response.
    
    Args:
        image: PIL Image to convert
        
    Returns:
        bytes: PNG-encoded image data
        
    Note:
        Includes cleanup of BytesIO buffer to prevent memory leaks
    """
    try:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()
    finally:
        img_byte_arr.close()
        del img_byte_arr
        gc.collect()

async def file_to_pil(file: UploadFile) -> Image.Image:
    """
    Converts uploaded file to PIL Image.
    
    Args:
        file: FastAPI UploadFile object
        
    Returns:
        PIL Image object
        
    Note:
        Handles format conversion to ensure compatibility
    """
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        # Convert to RGB if needed to ensure consistent format
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        return image
    finally:
        del contents
        gc.collect()

@app.post("/text-to-image")
async def text_to_image(request: TextToImageRequest):
    """
    Generate image from text prompt.
    
    Args:
        request: Text-to-image generation parameters
        
    Returns:
        StreamingResponse: Generated image in PNG format
        
    Raises:
        HTTPException: If generation fails
    """
    try:
        config = SDXLConfig(
            model=request.model_name,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance=request.guidance,
            seed=request.seed,
            scheduler_name=request.scheduler_name,
            loras=[LoRAConfig(name=lora.name, weight=lora.weight) for lora in request.loras]
        )
        
        output_image = sdxl.text_to_image(config)
        image_bytes = image_to_bytes(output_image)
        return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
    except Exception as e:
        handle_exception(e)
    finally:
        if 'output_image' in locals():
            del output_image
        if 'image_bytes' in locals():
            del image_bytes
        gc.collect()

@app.post("/image-to-image")
async def image_to_image(
    image: UploadFile = File(...),
    request: str = Form(...)
):
    """
    Transform existing image based on prompt.
    
    Args:
        image: Source image file
        request: JSON string containing transformation parameters
        
    Returns:
        StreamingResponse: Transformed image in PNG format
        
    Raises:
        HTTPException: If transformation fails or request format is invalid
    """
    try:
        request_dict = json.loads(request)
        request_data = ImageToImageRequest(**request_dict)
        
        source_image = await file_to_pil(image)
        
        config = SDXLConfig(
            model=request_data.model_name,
            prompt=request_data.prompt,
            negative_prompt=request_data.negative_prompt,
            source_image=source_image,
            strength=request_data.strength,
            steps=request_data.steps,
            guidance=request_data.guidance,
            seed=request_data.seed,
            scheduler_name=request_data.scheduler_name,
            loras=[LoRAConfig(name=lora.name, weight=lora.weight) for lora in request_data.loras]
        )
        
        output_image = sdxl.image_to_image(config)
        image_bytes = image_to_bytes(output_image)
        return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        handle_exception(e)
    finally:
        if 'source_image' in locals():
            del source_image
        if 'output_image' in locals():
            del output_image
        if 'image_bytes' in locals():
            del image_bytes
        gc.collect()

@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    request: str = Form(...)
):
    """
    Perform inpainting on masked area of image.
    
    Args:
        image: Source image file
        mask: Mask image file (black and white)
        request: JSON string containing inpainting parameters
        
    Returns:
        StreamingResponse: Inpainted image in PNG format
        
    Raises:
        HTTPException: If inpainting fails or request format is invalid
    """
    try:
        request_dict = json.loads(request)
        request_data = InpaintRequest(**request_dict)
        
        source_image = await file_to_pil(image)
        mask_image = await file_to_pil(mask)
        
        # Ensure mask is in grayscale format
        if mask_image.mode != "L":
            mask_image = mask_image.convert("L")
        
        config = SDXLConfig(
            model=request_data.model_name,
            prompt=request_data.prompt,
            negative_prompt=request_data.negative_prompt,
            source_image=source_image,
            mask_image=mask_image,
            steps=request_data.steps,
            guidance=request_data.guidance,
            seed=request_data.seed,
            scheduler_name=request_data.scheduler_name,
            loras=[LoRAConfig(name=lora.name, weight=lora.weight) for lora in request_data.loras]
        )
        
        output_image = sdxl.inpaint(config)
        image_bytes = image_to_bytes(output_image)
        return StreamingResponse(BytesIO(image_bytes), media_type="image/png")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        handle_exception(e)
    finally:
        if 'source_image' in locals():
            del source_image
        if 'mask_image' in locals():
            del mask_image
        if 'output_image' in locals():
            del output_image
        if 'image_bytes' in locals():
            del image_bytes
        gc.collect()

def handle_exception(e: Exception) -> None:
    """
    Standardized exception handling for API endpoints.
    
    Args:
        e: Exception to handle
        
    Raises:
        HTTPException: With appropriate status code and message
    """
    if isinstance(e, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(e))
    elif isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check API health status.
    
    Returns:
        dict: Status information
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)