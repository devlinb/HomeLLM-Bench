#!/usr/bin/env python3
"""
F5-TTS Persistent Service
A service wrapper that keeps the model loaded in memory for multiple generations.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

class F5TTSService:
    """
    Persistent F5-TTS service that keeps model loaded in memory.
    """
    
    def __init__(self, model_name: str = "F5TTS_v1_Base", device: str = "auto"):
        """
        Initialize the F5-TTS service.
        
        Args:
            model_name: Name of the F5-TTS model to use
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.vocoder = None
        self.is_loaded = False
        self.load_time = 0.0
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_model(self):
        """Load the F5-TTS model into memory."""
        if self.is_loaded:
            self.logger.info("Model already loaded")
            return
            
        self.logger.info(f"Loading F5-TTS model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Import F5-TTS modules
            from f5_tts.model import DiT, UNetT
            from f5_tts.infer.utils_infer import (
                load_model as load_f5_model,
                load_vocoder,
                preprocess_ref_audio_text,
                infer_process,
                remove_silence_for_generated_wav
            )
            import torch
            
            # Store the utility functions
            self.load_f5_model = load_f5_model
            self.load_vocoder = load_vocoder
            self.preprocess_ref_audio_text = preprocess_ref_audio_text
            self.infer_process = infer_process
            self.remove_silence_for_generated_wav = remove_silence_for_generated_wav
            
            # Set device first
            import torch
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model configuration
            from omegaconf import OmegaConf
            from hydra.utils import get_class
            from cached_path import cached_path
            from importlib.resources import files
            
            # Load model config
            config_path = str(files("f5_tts").joinpath(f"configs/{self.model_name}.yaml"))
            self.model_cfg = OmegaConf.load(config_path)
            
            # Get model class and architecture
            model_cls = get_class(f"f5_tts.model.{self.model_cfg.model.backbone}")
            model_arc = self.model_cfg.model.arch
            
            # Get checkpoint path
            repo_name, ckpt_step, ckpt_type = "F5-TTS", 1250000, "safetensors"
            
            # Override for different models
            if self.model_name == "F5TTS_Base":
                ckpt_step = 1200000
            elif self.model_name == "E2TTS_Base":
                repo_name = "E2-TTS"
                ckpt_step = 1200000
            
            ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{self.model_name}/model_{ckpt_step}.{ckpt_type}"))
            
            # Get vocab file path
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
            
            # Load the main model
            self.model = self.load_f5_model(
                model_cls=model_cls,
                model_cfg=model_arc,
                ckpt_path=ckpt_path,
                mel_spec_type="vocos",
                vocab_file=vocab_file,
                device=self.device
            )
            
            # Load vocoder
            self.vocoder = self.load_vocoder(device=self.device)
                
            self.load_time = time.time() - start_time
            self.is_loaded = True
            
            self.logger.info(f"Model loaded successfully in {self.load_time:.2f} seconds")
            self.logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_speech(
        self,
        text: str,
        output_path: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        remove_silence: bool = True,
        nfe_step: int = 32,
        cfg_strength: float = 2.0
    ) -> dict:
        """
        Generate speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the generated audio
            ref_audio: Optional reference audio file
            ref_text: Optional reference text
            speed: Speech speed multiplier
            remove_silence: Whether to remove silence
            nfe_step: Number of NFE steps
            cfg_strength: CFG strength
            
        Returns:
            Dictionary with generation metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        self.logger.info(f"Generating speech for text: {text[:50]}...")
        start_time = time.time()
        
        try:
            # Use default reference audio if none provided
            if ref_audio is None:
                ref_audio = "/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/f5_tts/infer/examples/basic/basic_ref_en.wav"
                ref_text = "Some call me nature, others call me mother nature."
            
            # Preprocess reference audio and text
            ref_audio_processed, ref_text_processed = self.preprocess_ref_audio_text(
                ref_audio, ref_text
            )
            
            # Generate speech
            result = self.infer_process(
                ref_audio=ref_audio_processed,
                ref_text=ref_text_processed,
                gen_text=text,
                model_obj=self.model,
                vocoder=self.vocoder,
                speed=speed,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                device=self.device
            )
            
            # Unpack the tuple (audio, sample_rate, spectrogram)
            generated_audio, actual_sample_rate, spectrogram = result
            
            # Save audio first
            import torchaudio
            import torch
            
            # Convert to tensor if needed
            if not isinstance(generated_audio, torch.Tensor):
                generated_audio = torch.from_numpy(generated_audio)
            
            torchaudio.save(
                output_path,
                generated_audio.unsqueeze(0),
                sample_rate=actual_sample_rate
            )
            
            # Remove silence if requested (operates on the saved file)
            if remove_silence:
                self.remove_silence_for_generated_wav(output_path)
            
            generation_time = time.time() - start_time
            
            # Calculate audio duration
            audio_duration = len(generated_audio) / actual_sample_rate
            rtf = generation_time / audio_duration
            
            metrics = {
                "generation_time": generation_time,
                "audio_duration": audio_duration,
                "rtf": rtf,
                "output_path": output_path,
                "text_length": len(text),
                "sample_rate": actual_sample_rate,
                "audio_samples": len(generated_audio)
            }
            
            self.logger.info(f"Generated {audio_duration:.2f}s audio in {generation_time:.2f}s (RTF: {rtf:.4f})")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get service statistics."""
        return {
            "is_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "load_time": self.load_time
        }
    
    def unload_model(self):
        """Unload the model from memory."""
        if self.is_loaded:
            self.logger.info("Unloading model...")
            self.model = None
            self.vocoder = None
            self.is_loaded = False
            
            # Clear GPU cache if using CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
                
            self.logger.info("Model unloaded")
    
    def __del__(self):
        """Cleanup when service is destroyed."""
        self.unload_model()


# Example usage and testing
if __name__ == "__main__":
    # Create service
    service = F5TTSService()
    
    # Load model
    service.load_model()
    
    # Generate speech
    metrics = service.generate_speech(
        text="Hello, this is a test of the F5-TTS service.",
        output_path="/tmp/test_service_output.wav"
    )
    
    print("Generation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    service.unload_model()