"""
Dynamic batching implementation for improved inference throughput.
Groups multiple requests together to maximize GPU utilization.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from collections import deque
from PIL import Image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicBatcher:
    """
    Dynamic batching system for image classification inference.
    
    Automatically groups incoming requests into batches to maximize
    GPU utilization while maintaining low latency.
    """
    
    def __init__(
        self,
        predictor,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,  # 100ms
        min_batch_size: int = 1,
    ):
        """
        Initialize dynamic batcher.
        
        Args:
            predictor: Model predictor instance
            max_batch_size: Maximum batch size to process
            max_wait_time: Maximum time to wait for batch to fill (seconds)
            min_batch_size: Minimum batch size before processing
        """
        self.predictor = predictor
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.min_batch_size = min_batch_size
        
        self.request_queue: deque = deque()
        self.pending_futures: Dict[int, asyncio.Future] = {}
        self.request_id_counter = 0
        self.is_processing = False
        self._process_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0
        self.avg_wait_time = 0.0
        
    async def start(self):
        """Start the background batch processing task."""
        if self._process_task is None:
            self._process_task = asyncio.create_task(self._process_loop())
            logger.info("Dynamic batcher started")
    
    async def stop(self):
        """Stop the batcher and cleanup."""
        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None
        logger.info("Dynamic batcher stopped")
    
    async def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        """
        Submit a prediction request to the batcher.
        
        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
        
        Returns:
            Prediction results dictionary
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        request_id = self.request_id_counter
        self.request_id_counter += 1
        self.total_requests += 1
        
        request = {
            'id': request_id,
            'image': image,
            'top_k': top_k,
            'timestamp': time.time(),
            'future': future,
        }
        
        self.request_queue.append(request)
        self.pending_futures[request_id] = future
        
        # Start processing if not already running
        if not self.is_processing and self._process_task is None:
            self.is_processing = True
            self._process_task = asyncio.create_task(self._process_loop())
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise
    
    async def _process_loop(self):
        """Main batch processing loop."""
        while True:
            try:
                # Wait for enough requests or timeout
                await self._wait_for_batch()
                
                if self.request_queue:
                    await self._process_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _wait_for_batch(self):
        """Wait until we have enough requests or timeout."""
        start_time = time.time()
        
        while len(self.request_queue) < self.min_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= self.max_wait_time:
                break
            await asyncio.sleep(0.01)
    
    async def _process_batch(self):
        """Process a batch of requests."""
        if not self.request_queue:
            return
        
        batch_start_time = time.time()
        batch = []
        
        while self.request_queue and len(batch) < self.max_batch_size:
            request = self.request_queue.popleft()
            batch.append(request)
        
        if not batch:
            return
        
        self.total_batches += 1
        
        current_avg = self.avg_batch_size
        n = self.total_batches
        self.avg_batch_size = ((n - 1) * current_avg + len(batch)) / n
        
        images = [req['image'] for req in batch]
        top_ks = [req['top_k'] for req in batch]
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.predictor.predict_batch(images, top_k=max(top_ks))
            )
            
            for i, request in enumerate(batch):
                wait_time = time.time() - request['timestamp']
                
                total_wait = self.avg_wait_time * (self.total_requests - len(batch))
                self.avg_wait_time = (total_wait + wait_time) / self.total_requests
                
                if not request['future'].done():
                    if isinstance(results, list) and i < len(results):
                        request['future'].set_result(results[i])
                    else:
                        request['future'].set_result(results)
                    
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            for request in batch:
                if not request['future'].done():
                    request['future'].set_exception(e)
        
        for request in batch:
            self.pending_futures.pop(request['id'], None)
        
        batch_time = time.time() - batch_start_time
        logger.debug(
            f"Processed batch of {len(batch)} images in {batch_time:.3f}s "
            f"(avg wait: {self.avg_wait_time*1000:.1f}ms)"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batcher performance metrics."""
        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'avg_batch_size': self.avg_batch_size,
            'avg_wait_time_ms': self.avg_wait_time * 1000,
            'queue_size': len(self.request_queue),
            'max_batch_size': self.max_batch_size,
            'efficiency': self.avg_batch_size / self.max_batch_size if self.max_batch_size > 0 else 0,
        }


class BatchPredictorAPI:
    """
    API wrapper that integrates dynamic batching with the existing API.
    """
    
    def __init__(self, predictor, config: Optional[Dict] = None):
        config = config or {}
        
        self.batcher = DynamicBatcher(
            predictor=predictor,
            max_batch_size=config.get('max_batch_size', 32),
            max_wait_time=config.get('max_wait_time', 0.1),
            min_batch_size=config.get('min_batch_size', 1),
        )
        
        self._started = False
    
    async def start(self):
        if not self._started:
            await self.batcher.start()
            self._started = True
    
    async def stop(self):
        if self._started:
            await self.batcher.stop()
            self._started = False
    
    async def predict(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        if not self._started:
            await self.start()
        
        return await self.batcher.predict(image, top_k=top_k)
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.batcher.get_metrics()


if __name__ == "__main__":
    import asyncio
    
    async def benchmark():
        class MockPredictor:
            def predict_batch(self, images, top_k=5):
                time.sleep(0.05)
                return [
                    {'predictions': [{'label': 'cat', 'confidence': 0.9}]}
                    for _ in images
                ]
        
        predictor = MockPredictor()
        batch_predictor = BatchPredictorAPI(
            predictor,
            config={'max_batch_size': 16, 'max_wait_time': 0.05}
        )
        
        test_images = [Image.new('RGB', (224, 224), color='red') for _ in range(50)]
        
        start = time.time()
        for img in test_images[:10]:
            await batch_predictor.predict(img)
        sequential_time = time.time() - start
        
        batch_predictor.batcher.total_requests = 0
        batch_predictor.batcher.total_batches = 0
        
        start = time.time()
        tasks = [batch_predictor.predict(img) for img in test_images[:10]]
        await asyncio.gather(*tasks)
        batched_time = time.time() - start
        
        metrics = batch_predictor.get_metrics()
        
        print(f"Sequential (10 requests): {sequential_time*1000:.1f}ms")
        print(f"Batched (10 requests): {batched_time*1000:.1f}ms")
        print(f"Speedup: {sequential_time/batched_time:.2f}x")
        print(f"Metrics: {metrics}")
    
    asyncio.run(benchmark())
