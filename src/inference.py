"""
Stable Diffusion XL (SDXL) Inference Module

This module provides a high-performance, memory-efficient implementation for running
SDXL models. It supports text-to-image, image-to-image, and inpainting operations
with various schedulers and LoRA models.

Key Features:
- Memory-efficient operation with aggressive cleanup
- Support for multiple scheduler types
- LoRA model integration
- Automatic device selection (CUDA/MPS/CPU)
- Comprehensive error handling

Example Usage:
    ```python
    from inference import SDXLInference, SDXLConfig, OperationType
    
    # Initialize inference engine
    sdxl = SDXLInference()
    
    # Create configuration
    config = SDXLConfig(
        model="model_name",
        prompt="a photo of a cat",
        width=768,
        height=768
    )
    
    # Generate image
    image = sdxl.text_to_image(config)
    ```

Note: This module is designed to run without persistent storage, making it
suitable for serverless environments like Cloud Run.
"""

from dataclasses import dataclass, field
from typing import Optional, Type, Union, List, Dict, Tuple
from enum import Enum
import torch
import random
import os
from PIL import Image
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    LMSDiscreteScheduler
)

# Initialize by clearing any existing cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
elif torch.backends.mps.is_available():
    torch.mps.empty_cache()

class OperationType(Enum):
    """
    Defines the types of operations supported by the SDXL inference engine.
    
    Attributes:
        TEXT_TO_IMAGE: Generate image from text prompt
        IMAGE_TO_IMAGE: Transform existing image based on prompt
        INPAINT: Modify specific parts of image using mask
    """
    TEXT_TO_IMAGE = "txt2img"
    IMAGE_TO_IMAGE = "img2img"
    INPAINT = "inpaint"

# Mapping of scheduler names to their implementations
SCHEDULER_DICT = {
    'Euler': EulerDiscreteScheduler,
    'Euler Karras': EulerDiscreteScheduler,
    'Euler Ancestral': EulerAncestralDiscreteScheduler,
    'Heun': HeunDiscreteScheduler,
    'Heun Karras': HeunDiscreteScheduler,
    'DPM++ 2M': DPMSolverMultistepScheduler,
    'DPM++ 2M Karras': DPMSolverMultistepScheduler,
    'DPM++ 2M SDE': DPMSolverMultistepScheduler,
    'DPM++ 2M SDE Karras': DPMSolverMultistepScheduler,
    'K DPM 2': KDPM2DiscreteScheduler,
    'K DPM 2 Karras': KDPM2DiscreteScheduler,
    'K DPM 2 Ancestral': KDPM2AncestralDiscreteScheduler,
    'UniPC 2M': UniPCMultistepScheduler,
    'UniPC 2M Karras': UniPCMultistepScheduler,
    'LMS': LMSDiscreteScheduler,
    'LMS Karras': LMSDiscreteScheduler
}

@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) models.
    
    Attributes:
        name (str): Name of the LoRA model file (without extension)
        weight (float): Weight to apply to the LoRA model (-2.0 to 2.0)
    
    Example:
        ```python
        lora = LoRAConfig(name="style_lora", weight=0.7)
        ```
    """
    name: str
    weight: float = 1.0
    
    def validate(self):
        """
        Validates LoRA configuration parameters.
        
        Raises:
            ValueError: If weight is outside valid range or not in 0.1 steps
        """
        if not -2 <= self.weight <= 2:
            raise ValueError(f"LoRA weight must be between -2 and 2, got {self.weight}")
        if round(self.weight * 10) != self.weight * 10:
            raise ValueError(f"LoRA weight must be in steps of 0.1, got {self.weight}")

@dataclass
class SDXLConfig:
    """
    Configuration for SDXL inference operations.
    
    Attributes:
        model (str): Name of the main model file (without extension)
        scheduler_name (str): Name of the scheduler to use
        prompt (str): Main prompt describing desired image
        negative_prompt (str): Prompt describing elements to avoid
        width (int): Output image width in pixels
        height (int): Output image height in pixels
        steps (int): Number of inference steps
        guidance (float): Classifier-free guidance scale
        seed (Optional[int]): Random seed for reproducibility
        source_image (Optional[Image.Image]): Source image for img2img/inpaint
        strength (Optional[float]): Transformation strength for img2img
        mask_image (Optional[Image.Image]): Mask image for inpainting
        loras (List[LoRAConfig]): List of LoRA models to apply
    
    Example:
        ```python
        config = SDXLConfig(
            model="realistic_v1",
            prompt="a photo of a cat",
            width=768,
            height=768,
            steps=30,
            guidance=7.0
        )
        ```
    """
    model: str
    scheduler_name: str = "DPM++ 2M"
    
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 768
    height: int = 1024
    steps: int = 30
    guidance: float = 7.0
    seed: Optional[int] = None
    
    source_image: Optional[Image.Image] = None
    strength: Optional[float] = None
    
    mask_image: Optional[Image.Image] = None

    loras: List[LoRAConfig] = field(default_factory=list)

    def validate_for_operation(self, operation_type: OperationType):
        """
        Validates configuration for specific operation type.
        
        Args:
            operation_type: Type of operation to validate for
            
        Raises:
            ValueError: If configuration is invalid for operation
        """
        if not self.prompt:
            raise ValueError("Prompt is required for all operations")
            
        if operation_type == OperationType.IMAGE_TO_IMAGE:
            if not self.source_image:
                raise ValueError("source_image is required for img2img operation")
            if self.strength is None:
                raise ValueError("strength is required for img2img operation")
                
        elif operation_type == OperationType.INPAINT:
            if not self.source_image:
                raise ValueError("source_image is required for inpaint operation")
            if not self.mask_image:
                raise ValueError("mask_image is required for inpaint operation")
        
        if self.scheduler_name not in SCHEDULER_DICT:
            raise ValueError(f"Invalid scheduler name. Must be one of: {', '.join(SCHEDULER_DICT.keys())}")
        
        for lora in self.loras:
            lora.validate()

class DeviceManager:
    """
    Manages device selection and configuration for inference.
    Automatically selects the best available device (CUDA > MPS > CPU)
    and configures appropriate precision.
    """
    @staticmethod
    def get_device_config() -> Tuple[str, torch.dtype]:
        """
        Determines optimal device and precision configuration.
        
        Returns:
            Tuple[str, torch.dtype]: Device name and precision dtype
        """
        if torch.cuda.is_available():
            return "cuda", torch.float16
        elif torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

class SchedulerManager:
    """
    Manages scheduler configuration for different inference methods.
    Handles special cases like Karras sigmas and SDE configurations.
    """
    @staticmethod
    def configure_scheduler(pipeline, scheduler_name: str):
        """
        Configures scheduler with appropriate parameters.
        
        Args:
            pipeline: SDXL pipeline instance
            scheduler_name: Name of scheduler to configure
            
        Returns:
            Configured scheduler instance
            
        Raises:
            ValueError: If scheduler name is invalid
        """
        if scheduler_name not in SCHEDULER_DICT:
            raise ValueError(f"Invalid scheduler name: {scheduler_name}")
        
        scheduler_class = SCHEDULER_DICT[scheduler_name]
        scheduler_config = pipeline.scheduler.config
        
        # Configure special cases
        if scheduler_name in ['Euler Karras', 'Heun Karras', 'K DPM 2 Karras']:
            return scheduler_class.from_config(scheduler_config, use_karras_sigmas=True)
        elif scheduler_name == 'DPM++ 2M SDE':
            return scheduler_class.from_config(scheduler_config, algorithm_type="sde-dpmsolver++")
        elif scheduler_name == 'DPM++ 2M SDE Karras':
            return scheduler_class.from_config(scheduler_config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        elif scheduler_name in ['DPM++ 2M Karras', 'UniPC 2M Karras', 'LMS Karras']:
            return scheduler_class.from_config(scheduler_config, use_karras_sigmas=True)
        else:
            return scheduler_class.from_config(scheduler_config)

class SDXLPipelineFactory:
    """
    Factory class for creating and configuring SDXL pipelines.
    Handles model loading, LoRA integration, and scheduler setup.
    """
    @staticmethod
    def create_pipeline(operation_type: OperationType, model_name: str, scheduler_name: str, loras: List[LoRAConfig] = None):
        """
        Creates and configures an SDXL pipeline for specific operation.
        
        Args:
            operation_type: Type of operation (text2img, img2img, inpaint)
            model_name: Name of main model file
            scheduler_name: Name of scheduler to use
            loras: Optional list of LoRA configurations
            
        Returns:
            Configured pipeline instance
            
        Raises:
            FileNotFoundError: If model or LoRA files not found
        """
        # Verify model exists
        model_path = f"models/main-models/{model_name}.safetensors"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_name}.safetensors")
            
        # Select appropriate pipeline class
        pipeline_map = {
            OperationType.TEXT_TO_IMAGE: StableDiffusionXLPipeline,
            OperationType.IMAGE_TO_IMAGE: StableDiffusionXLImg2ImgPipeline,
            OperationType.INPAINT: StableDiffusionXLInpaintPipeline
        }
        
        pipeline_class = pipeline_map[operation_type]
        pipeline = pipeline_class.from_single_file(
            model_path,
            use_safetensors=True,
            variant="fp16",
            requires_safety_checker=False,
            safety_checker=None
        )
        
        # Load and configure LoRA models if specified
        if loras:
            for lora in loras:
                lora_path = f"models/lora-models/{lora.name}.safetensors"
                if not os.path.exists(lora_path):
                    raise FileNotFoundError(f"LoRA file not found: {lora.name}.safetensors")
                pipeline.load_lora_weights(
                    lora_path,
                    adapter_name=lora.name
                )
            
            pipeline.set_adapters(
                adapter_names=[lora.name for lora in loras],
                adapter_weights=[lora.weight for lora in loras]
            )
        
        # Configure scheduler
        pipeline.scheduler = SchedulerManager.configure_scheduler(pipeline, scheduler_name)
        return pipeline

class CompelWrapper:
    """
    Wrapper for Compel prompt processing system.
    Handles creation of text embeddings for inference.
    """
    @staticmethod
    def create_embeddings(pipeline, prompt: str, negative_prompt: str):
        """
        Creates embeddings from prompts using Compel.
        
        Args:
            pipeline: SDXL pipeline instance
            prompt: Main prompt text
            negative_prompt: Negative prompt text
            
        Returns:
            Tuple of prompt embeddings and pooled embeddings
        """
        compel = Compel(
            tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
            text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False
        )
        embeddings = compel([prompt, negative_prompt])
        del compel  # Explicitly delete to free memory
        return embeddings

class SDXLInference:
    """
    Main inference class for SDXL operations.
    Handles all types of image generation with memory-efficient operation.
    
    Example:
        ```python
        inference = SDXLInference()
        config = SDXLConfig(...)
        image = inference.text_to_image(config)
        ```
    """
    def __init__(self):
        """Initialize inference engine with optimal device configuration."""
        self.device, self.precision = DeviceManager.get_device_config()
    
    def _setup_pipeline(self, config: SDXLConfig, operation_type: OperationType):
        """
        Sets up pipeline for inference operation.
        
        Args:
            config: Operation configuration
            operation_type: Type of operation
            
        Returns:
            Configured pipeline instance
        """
        self._clear_memory()
        
        pipeline = SDXLPipelineFactory.create_pipeline(
            operation_type,
            config.model,
            config.scheduler_name,
            config.loras
        ).to(device=self.device, dtype=self.precision)
        
        return pipeline
    
    def _get_generator(self, seed: Optional[int] = None) -> Tuple[torch.Generator, int]:
        """
        Creates random number generator with seed.
        
        Args:
            seed: Optional seed value
            
        Returns:
            Tuple of generator and actual seed used
        """
        actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        return torch.Generator(device=self.device).manual_seed(actual_seed), actual_seed
    
    def _log_operation_details(self, config: SDXLConfig, operation_type: OperationType, actual_values: dict):
        """
        Logs details of inference operation for monitoring.
        
        Args:
            config: Operation configuration
            operation_type: Type of operation
            actual_values: Dictionary of actual values used
        """
        print(f"Operation: {operation_type.value}")
        print(f"Device: {self.device}")
        print(f"Precision: {self.precision}")
        print(f"Model: {config.model}")
        print(f"Scheduler: {config.scheduler_name}")
        print(f"Prompt: {config.prompt}")
        print(f"Negative Prompt: {config.negative_prompt}")
        print(f"Width: {actual_values.get('width', config.width)}")
        print(f"Height: {actual_values.get('height', config.height)}")
        print(f"Steps: {config.steps}")
        print(f"Guidance Scale: {config.guidance}")
        print(f"Seed: {actual_values['seed']}")

        if config.loras:
            print("LoRAs:")
            for lora in config.loras:
                print(f"  - {lora.name} (weight: {lora.weight})")
            
    def text_to_image(self, config: SDXLConfig) -> Image.Image:
        """
        Generates image from text prompt.
        
        Args:
            config: Generation configuration
            
        Returns:
            Generated PIL Image
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config.validate_for_operation(OperationType.TEXT_TO_IMAGE)
            pipeline = self._setup_pipeline(config, OperationType.TEXT_TO_IMAGE)
            
            # Create text embeddings
            conditioning, pooled = CompelWrapper.create_embeddings(
                pipeline, config.prompt, config.negative_prompt
            )
            
            generator, actual_seed = self._get_generator(config.seed)
            actual_values = {
                'seed': actual_seed,
                'width': config.width,
                'height': config.height
            }
            
            self._log_operation_details(config, OperationType.TEXT_TO_IMAGE, actual_values)
            
            # Generate image
            output = pipeline(
                prompt_embeds=conditioning[0:1],
                negative_prompt_embeds=conditioning[1:2],
                pooled_prompt_embeds=pooled[0:1],
                negative_pooled_prompt_embeds=pooled[1:2],
                num_inference_steps=config.steps,
                guidance_scale=config.guidance,
                width=config.width,
                height=config.height,
                generator=generator
            )
            
            result_image = output.images[0]
            del output
            return result_image
        finally:
            self._clear_memory()
    
    def image_to_image(self, config: SDXLConfig) -> Image.Image:
        """
        Transforms source image based on prompt.
        
        Args:
            config: Transformation configuration
            
        Returns:
            Transformed PIL Image
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config.validate_for_operation(OperationType.IMAGE_TO_IMAGE)
            init_image = config.source_image
            width, height = init_image.size
            
            pipeline = self._setup_pipeline(config, OperationType.IMAGE_TO_IMAGE)
            conditioning, pooled = CompelWrapper.create_embeddings(
                pipeline, config.prompt, config.negative_prompt
            )
            
            generator, actual_seed = self._get_generator(config.seed)
            actual_values = {
                'seed': actual_seed,
                'width': width,
                'height': height
            }
            
            self._log_operation_details(config, OperationType.IMAGE_TO_IMAGE, actual_values)
            
            # Transform image
            output = pipeline(
                prompt_embeds=conditioning[0:1],
                negative_prompt_embeds=conditioning[1:2],
                pooled_prompt_embeds=pooled[0:1],
                negative_pooled_prompt_embeds=pooled[1:2],
                image=init_image,
                strength=config.strength,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance,
                width=width,
                height=height,
                generator=generator
            )
            
            result_image = output.images[0]
            del output
            return result_image
        finally:
            self._clear_memory()
    
    def inpaint(self, config: SDXLConfig) -> Image.Image:
        """
        Performs inpainting on masked area of image.
        
        Args:
            config: Inpainting configuration
            
        Returns:
            Inpainted PIL Image
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config.validate_for_operation(OperationType.INPAINT)
            init_image = config.source_image
            width, height = init_image.size
            mask_image = config.mask_image
            
            pipeline = self._setup_pipeline(config, OperationType.INPAINT)
            conditioning, pooled = CompelWrapper.create_embeddings(
                pipeline, config.prompt, config.negative_prompt
            )
            
            generator, actual_seed = self._get_generator(config.seed)
            actual_values = {
                'seed': actual_seed,
                'width': width,
                'height': height
            }
            
            self._log_operation_details(config, OperationType.INPAINT, actual_values)
            
            # Perform inpainting
            output = pipeline(
                prompt_embeds=conditioning[0:1],
                negative_prompt_embeds=conditioning[1:2],
                pooled_prompt_embeds=pooled[0:1],
                negative_pooled_prompt_embeds=pooled[1:2],
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=config.steps,
                guidance_scale=config.guidance,
                width=width,
                height=height,
                generator=generator
            )
            
            result_image = output.images[0]
            del output
            return result_image
        finally:
            self._clear_memory()
    
    def _clear_memory(self):
        """
        Performs thorough cleanup of memory and resources.
        Ensures no memory leaks or cache buildup over time.
        """
        # Clear pipeline
        if hasattr(self, 'pipeline'):
            del self.pipeline
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Clear MPS cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear any remaining tensors
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except:
                pass