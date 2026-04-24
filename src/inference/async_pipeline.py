"""
Async Inference Pipeline with Non-blocking I/O and Pipeline Parallelism.

This module implements a high-performance inference pipeline that overlaps:
1. Data Loading & Preprocessing (CPU)
2. Model Inference (GPU)
3. Post-processing & Response Formatting (CPU)

Features:
- AsyncIO-based non-blocking operations
- Bounded queues for backpressure control
- Dynamic batching integration
- Graceful shutdown handling
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import torch
from PIL import Image
import io

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class PipelineStats:
    """Real-time statistics for the pipeline."""
    requests_processed: int = 0
    avg_latency_ms: float = 0.0
    queue_depth: int = 0
    gpu_utilization: float = 0.0
    errors: int = 0
    start_time: float = 0.0

    def __post_init__(self):
        if self.start_time == 0.0:
            self.start_time = time.time()


class AsyncInferencePipeline:
    """
    High-performance async inference pipeline.
    
    Usage:
        pipeline = AsyncInferencePipeline(model, batch_size=32, max_queue_size=100)
        await pipeline.start()
        result = await pipeline.predict(image_bytes)
        await pipeline.stop()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 32,
        max_queue_size: int = 100,
        timeout_seconds: float = 5.0,
        enable_profiling: bool = True
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.timeout_seconds = timeout_seconds
        self.enable_profiling = enable_profiling
        
        # Pipeline queues
        self.preprocess_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.inference_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.postprocess_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Results storage for async correlation
        self.pending_results: Dict[str, asyncio.Future] = {}
        
        # Statistics
        self.stats = PipelineStats()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Locks for thread safety
        self._stats_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        
        logger.info(f"Initialized AsyncInferencePipeline on {device} with batch_size={batch_size}")

    async def start(self):
        """Start all pipeline stages as background tasks."""
        if self._running:
            raise RuntimeError("Pipeline is already running")
            
        self._running = True
        self.stats.start_time = time.time()
        
        # Start worker tasks
        self._tasks = [
            asyncio.create_task(self._preprocess_worker(), name="PreprocessWorker"),
            asyncio.create_task(self._inference_worker(), name="InferenceWorker"),
            asyncio.create_task(self._postprocess_worker(), name="PostprocessWorker"),
            asyncio.create_task(self._stats_collector(), name="StatsCollector"),
        ]
        
        logger.info("AsyncInferencePipeline started with 4 worker tasks")

    async def stop(self):
        """Gracefully shutdown the pipeline."""
        self._running = False
        
        # Signal workers to stop
        await self.preprocess_queue.put(None)
        await self.inference_queue.put(None)
        await self.postprocess_queue.put(None)
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        logger.info("AsyncInferencePipeline stopped")

    async def predict(self, image_data: bytes, request_id: str) -> Dict[str, Any]:
        """
        Submit an image for prediction and wait for result.
        
        Args:
            image_data: Raw image bytes
            request_id: Unique identifier for tracing
            
        Returns:
            Prediction result dictionary
        """
        if not self._running:
            raise RuntimeError("Pipeline is not running")
            
        if self.preprocess_queue.full():
            logger.warning(f"Queue full, rejecting request {request_id}")
            async with self._stats_lock:
                self.stats.errors += 1
            raise asyncio.QueueFull("Pipeline queue is full")
        
        # Create future for this request
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.pending_results[request_id] = future
        
        try:
            # Submit to pipeline
            await self.preprocess_queue.put((request_id, image_data, time.time()))
            
            # Wait for result with timeout
            result = await asyncio.wait_for(future, timeout=self.timeout_seconds)
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out")
            async with self._stats_lock:
                self.stats.errors += 1
            raise
        finally:
            self.pending_results.pop(request_id, None)

    async def _preprocess_worker(self):
        """Stage 1: Decode and preprocess images (CPU-bound, run in executor)."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            item = await self.preprocess_queue.get()
            if item is None:
                break
                
            request_id, image_data, submit_time = item
            
            try:
                # Offload CPU-bound preprocessing to thread pool
                preprocessed = await loop.run_in_executor(
                    None, 
                    self._preprocess_image, 
                    image_data
                )
                
                await self.inference_queue.put((request_id, preprocessed, submit_time))
                
            except Exception as e:
                logger.error(f"Preprocessing error for {request_id}: {e}")
                await self._set_result_error(request_id, f"Preprocessing failed: {str(e)}")

    async def _inference_worker(self):
        """Stage 2: Run model inference (GPU-bound)."""
        # Collect items for batching
        batch_buffer: List[Tuple[str, torch.Tensor, float]] = []
        batch_timer = 0.0
        
        while self._running:
            try:
                # Wait for item with short timeout to allow batching
                item = await asyncio.wait_for(
                    self.inference_queue.get(), 
                    timeout=0.01  # Small delay to accumulate batch
                )
                
                if item is None:
                    break
                    
                request_id, tensor, submit_time = item
                batch_buffer.append((request_id, tensor, submit_time))
                
                # Process batch if full or timeout reached
                if len(batch_buffer) >= self.batch_size:
                    await self._process_batch(batch_buffer)
                    batch_buffer = []
                    
            except asyncio.TimeoutError:
                # Timeout reached, process whatever we have
                if batch_buffer:
                    await self._process_batch(batch_buffer)
                    batch_buffer = []
                continue

    async def _process_batch(self, batch: List[Tuple[str, torch.Tensor, float]]):
        """Execute model inference on a batch."""
        if not batch:
            return
            
        request_ids = [item[0] for item in batch]
        tensors = torch.stack([item[1] for item in batch]).to(self.device)
        submit_times = [item[2] for item in batch]
        
        try:
            async with self._model_lock:
                with torch.inference_mode():
                    # Run inference
                    outputs = self.model(tensors)
                    
                    # Move to CPU for post-processing
                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu()
                        
            # Send to post-processing
            for i, request_id in enumerate(request_ids):
                output = outputs[i] if isinstance(outputs, list) else outputs[i]
                await self.postprocess_queue.put((
                    request_id, 
                    output, 
                    submit_times[i]
                ))
                
        except Exception as e:
            logger.error(f"Inference error for batch {request_ids}: {e}")
            for request_id in request_ids:
                await self._set_result_error(request_id, f"Inference failed: {str(e)}")

    async def _postprocess_worker(self):
        """Stage 3: Convert outputs to JSON responses (CPU-bound)."""
        loop = asyncio.get_event_loop()
        
        while self._running:
            item = await self.postprocess_queue.get()
            if item is None:
                break
                
            request_id, output, submit_time = item
            
            try:
                # Offload post-processing
                result = await loop.run_in_executor(
                    None,
                    self._postprocess_output,
                    output
                )
                
                # Add timing metadata
                total_latency = (time.time() - submit_time) * 1000
                result["latency_ms"] = total_latency
                
                await self._set_result_success(request_id, result)
                
                # Update stats
                async with self._stats_lock:
                    self.stats.requests_processed += 1
                    # Running average
                    n = self.stats.requests_processed
                    self.stats.avg_latency_ms = (
                        (self.stats.avg_latency_ms * (n - 1) + total_latency) / n
                    )
                    
            except Exception as e:
                logger.error(f"Postprocessing error for {request_id}: {e}")
                await self._set_result_error(request_id, f"Postprocessing failed: {str(e)}")

    def _preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image: decode, resize, normalize."""
        # Simulated preprocessing (replace with actual transforms)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        # Resize to model input size (e.g., 224x224)
        image = image.resize((224, 224))
        
        # Convert to tensor and normalize
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)

    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """Convert model output to prediction dictionary."""
        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=0)
        
        # Get top-5 predictions
        top_probs, top_indices = torch.topk(probs, 5)
        
        predictions = [
            {"class_id": idx.item(), "confidence": prob.item()}
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return {
            "predictions": predictions,
            "status": "success"
        }

    async def _set_result_success(self, request_id: str, result: Dict[str, Any]):
        """Set successful result for a request."""
        future = self.pending_results.get(request_id)
        if future and not future.done():
            future.set_result(result)

    async def _set_result_error(self, request_id: str, error_msg: str):
        """Set error result for a request."""
        future = self.pending_results.get(request_id)
        if future and not future.done():
            future.set_result({"status": "error", "message": error_msg})

    async def _stats_collector(self):
        """Periodically collect GPU utilization and other metrics."""
        while self._running:
            await asyncio.sleep(5.0)  # Collect every 5 seconds
            
            if torch.cuda.is_available():
                self.stats.gpu_utilization = torch.cuda.utilization(self.device)
            
            self.stats.queue_depth = (
                self.preprocess_queue.qsize() +
                self.inference_queue.qsize() +
                self.postprocess_queue.qsize()
            )
            
            if self.enable_profiling:
                logger.debug(
                    f"Pipeline Stats: processed={self.stats.requests_processed}, "
                    f"avg_latency={self.stats.avg_latency_ms:.2f}ms, "
                    f"queue_depth={self.stats.queue_depth}, "
                    f"gpu_util={self.stats.gpu_utilization:.1f}%"
                )

    def get_stats(self) -> PipelineStats:
        """Get current pipeline statistics."""
        return self.stats
