"""
Model Compression Pipeline

This module provides tools for compressing deep learning models for edge deployment.
Supports quantization, pruning, and conversion to TensorFlow Lite.
"""

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCompressor:
    """
    Compress PyTorch models for edge deployment.
    
    Supports:
    - Dynamic quantization
    - Static quantization
    - Pruning
    - TensorFlow Lite conversion
    """
    
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def dynamic_quantization(
        self,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            dtype: Quantization data type
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying dynamic quantization to {self.model_name}")
        
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.Conv2d, nn.Conv1d},
            dtype=dtype
        )
        
        return quantized_model
    
    def static_quantization(
        self,
        calibration_data: torch.utils.data.DataLoader,
        backend: str = 'fbgemm'
    ) -> nn.Module:
        """
        Apply static quantization with calibration.
        
        Args:
            calibration_data: Data loader for calibration
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
            
        Returns:
            Quantized model
        """
        logger.info(f"Applying static quantization to {self.model_name}")
        
        # Set backend
        torch.backends.quantized.engine = backend
        
        # Prepare model
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate
        logger.info("Calibrating model...")
        with torch.no_grad():
            for batch in calibration_data:
                if isinstance(batch, (list, tuple)):
                    self.model(*batch)
                else:
                    self.model(batch)
        
        # Convert
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return quantized_model
    
    def prune_model(
        self,
        amount: float = 0.3,
        method: str = 'l1_unstructured'
    ) -> nn.Module:
        """
        Prune model weights.
        
        Args:
            amount: Fraction of weights to prune
            method: Pruning method
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        logger.info(f"Pruning {amount*100}% of weights from {self.model_name}")
        
        # Prune all Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if method == 'l1_unstructured':
                    prune.l1_unstructured(module, name='weight', amount=amount)
                elif method == 'random_unstructured':
                    prune.random_unstructured(module, name='weight', amount=amount)
                
                # Make pruning permanent
                prune.remove(module, 'weight')
        
        return self.model
    
    def measure_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Measure model size and parameter count.
        
        Args:
            model: Model to measure
            
        Returns:
            Dictionary with size metrics
        """
        # Save to temporary file
        temp_path = Path(f"temp_{self.model_name}.pth")
        torch.save(model.state_dict(), temp_path)
        
        size_mb = temp_path.stat().st_size / (1024 * 1024)
        temp_path.unlink()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "size_mb": size_mb,
            "total_params": total_params,
            "trainable_params": trainable_params
        }
    
    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark model inference time.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of inference runs
            
        Returns:
            Benchmark results
        """
        import time
        
        model.eval()
        dummy_input = torch.randn(*input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(dummy_input)
                times.append(time.time() - start)
        
        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000
        }


class TFLiteConverter:
    """
    Convert PyTorch models to TensorFlow Lite format.
    """
    
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        
    def convert_to_tflite(
        self,
        input_shape: Tuple,
        output_path: str,
        quantize: bool = True,
        optimization: str = 'default'  # 'default', 'size', 'latency'
    ) -> str:
        """
        Convert PyTorch model to TensorFlow Lite.
        
        Args:
            input_shape: Input tensor shape
            output_path: Path to save TFLite model
            quantize: Apply quantization
            optimization: Optimization strategy
            
        Returns:
            Path to saved TFLite model
        """
        logger.info(f"Converting {self.model_name} to TensorFlow Lite")
        
        # Export to ONNX first
        onnx_path = f"temp_{self.model_name}.onnx"
        dummy_input = torch.randn(*input_shape)
        
        self.model.eval()
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Convert ONNX to TensorFlow
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_model_path = f"temp_{self.model_name}_tf"
        tf_rep.export_graph(tf_model_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Set optimization
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if optimization == 'size':
                converter.target_spec.supported_types = [tf.float16]
            elif optimization == 'latency':
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)
        
        # Cleanup
        Path(onnx_path).unlink()
        import shutil
        shutil.rmtree(tf_model_path)
        
        logger.info(f"TFLite model saved to {output_path}")
        logger.info(f"Model size: {output_path.stat().st_size / 1024:.2f} KB")
        
        return str(output_path)
    
    def benchmark_tflite(
        self,
        tflite_path: str,
        input_shape: Tuple,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark TFLite model inference.
        
        Args:
            tflite_path: Path to TFLite model
            input_shape: Input tensor shape
            num_runs: Number of runs
            
        Returns:
            Benchmark results
        """
        import time
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            times.append(time.time() - start)
        
        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000
        }


def compress_audio_model(
    model_path: str,
    output_dir: str,
    calibration_data: Optional[torch.utils.data.DataLoader] = None
) -> Dict[str, str]:
    """
    Compress audio detection model for edge deployment.
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory for compressed models
        calibration_data: Calibration data for quantization
        
    Returns:
        Dictionary of compressed model paths
    """
    from models.audio_detector import AudioDeepfakeDetector, LightweightAudioDetector
    
    logger.info("Compressing audio detection model")
    
    # Load model
    model = AudioDeepfakeDetector()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compressor = ModelCompressor(model, "audio_detector")
    
    # Original model metrics
    original_metrics = compressor.measure_model_size(model)
    logger.info(f"Original model: {original_metrics['size_mb']:.2f} MB")
    
    # Dynamic quantization
    quantized_model = compressor.dynamic_quantization()
    quantized_path = output_dir / "audio_detector_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_path)
    
    quantized_metrics = compressor.measure_model_size(quantized_model)
    logger.info(f"Quantized model: {quantized_metrics['size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {original_metrics['size_mb'] / quantized_metrics['size_mb']:.2f}x")
    
    # Convert to TFLite
    converter = TFLiteConverter(quantized_model, "audio_detector")
    tflite_path = converter.convert_to_tflite(
        input_shape=(1, 16000 * 10),  # 10 seconds at 16kHz
        output_path=str(output_dir / "audio_detector.tflite"),
        quantize=True,
        optimization='size'
    )
    
    return {
        "quantized_pytorch": str(quantized_path),
        "tflite": tflite_path
    }


def compress_video_model(
    model_path: str,
    output_dir: str
) -> Dict[str, str]:
    """
    Compress video detection model for edge deployment.
    
    Args:
        model_path: Path to trained model
        output_dir: Output directory for compressed models
        
    Returns:
        Dictionary of compressed model paths
    """
    from models.video_detector import VideoDeepfakeDetector, LightweightVideoDetector
    
    logger.info("Compressing video detection model")
    
    # Use lightweight model for edge
    model = LightweightVideoDetector(num_frames=15)
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compressor = ModelCompressor(model, "video_detector")
    
    # Quantize
    quantized_model = compressor.dynamic_quantization()
    quantized_path = output_dir / "video_detector_quantized.pth"
    torch.save(quantized_model.state_dict(), quantized_path)
    
    # Convert to TFLite
    converter = TFLiteConverter(quantized_model, "video_detector")
    tflite_path = converter.convert_to_tflite(
        input_shape=(1, 15, 3, 224, 224),
        output_path=str(output_dir / "video_detector.tflite"),
        quantize=True,
        optimization='size'
    )
    
    return {
        "quantized_pytorch": str(quantized_path),
        "tflite": tflite_path
    }


if __name__ == "__main__":
    # Example usage
    from models.audio_detector import LightweightAudioDetector
    
    # Create and compress a lightweight model
    model = LightweightAudioDetector()
    compressor = ModelCompressor(model, "test_audio")
    
    # Measure original
    metrics = compressor.measure_model_size(model)
    print(f"Original: {metrics}")
    
    # Quantize
    quantized = compressor.dynamic_quantization()
    quantized_metrics = compressor.measure_model_size(quantized)
    print(f"Quantized: {quantized_metrics}")
    
    # Benchmark
    bench = compressor.benchmark_inference(
        model,
        input_shape=(1, 128, 1000),
        num_runs=50
    )
    print(f"Inference time: {bench['mean_ms']:.2f} ± {bench['std_ms']:.2f} ms")
