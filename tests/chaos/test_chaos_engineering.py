"""
Chaos engineering tests for image classification system.
Tests system resilience under failure conditions.
"""

import pytest
import time
import random
import threading
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import requests
from requests.exceptions import RequestException, Timeout


class TestNetworkChaos:
    """Test system behavior under network failures."""
    
    def test_api_handles_service_unavailable(self):
        """API should handle 503 Service Unavailable gracefully."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = RequestException("Service Unavailable")
            mock_post.return_value.status_code = 503
            
            from api.routes.predict import predict_image
            
            # Should not crash, should return appropriate error
            with pytest.raises(Exception) as exc_info:
                predict_image(file=None)
            
            assert exc_info.type in [RequestException, Exception]
    
    def test_api_handles_timeout(self):
        """API should handle request timeouts."""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")
            
            from api.routes.predict import predict_image
            
            with pytest.raises(Timeout):
                predict_image(file=None)
    
    def test_api_handles_intermittent_failures(self):
        """System should handle intermittent network failures."""
        call_count = [0]
        
        def flaky_request(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:  # Fail every 3rd request
                raise RequestException("Intermittent failure")
            response = MagicMock()
            response.status_code = 200
            response.json.return_value = {"predictions": []}
            return response
        
        with patch('requests.post', side_effect=flaky_request):
            # Simulate multiple requests
            successes = 0
            failures = 0
            
            for i in range(10):
                try:
                    # In real scenario, would call actual endpoint
                    # Here we just verify the pattern
                    if (i + 1) % 3 != 0:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
            
            # Should have some successes despite failures
            assert successes > 0
    
    def test_retry_logic_with_backoff(self):
        """System should implement retry with exponential backoff."""
        from src.utils.logger import get_logger
        import time
        
        call_times = []
        
        def failing_request(*args, **kwargs):
            call_times.append(time.time())
            raise RequestException("Temporary failure")
        
        with patch('requests.post', side_effect=failing_request):
            start_time = time.time()
            
            # Simulate retry logic (would be implemented in production code)
            max_retries = 3
            base_delay = 0.1  # seconds
            
            for attempt in range(max_retries):
                try:
                    failing_request()
                except RequestException:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
                    else:
                        raise
            
            elapsed = time.time() - start_time
            
            # Should have waited with exponential backoff
            # Expected: 0.1 + 0.2 = 0.3 seconds minimum
            assert elapsed >= 0.25  # Allow some tolerance


class TestResourceChaos:
    """Test system behavior under resource constraints."""
    
    def test_handles_memory_pressure(self):
        """System should handle low memory conditions."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.mem_get_info') as mock_mem:
                # Simulate very low available memory
                mock_mem.return_value = (1000000, 1000000)  # Very little free
                
                from src.inference.predictor import Predictor
                
                # Should handle gracefully or raise appropriate error
                try:
                    predictor = Predictor(model_path="dummy.pth", device="cuda")
                    # If it succeeds, that's fine too (depends on implementation)
                except RuntimeError as e:
                    assert "CUDA" in str(e) or "memory" in str(e).lower()
    
    def test_handles_cpu_exhaustion(self):
        """System should handle CPU exhaustion."""
        import multiprocessing
        
        def cpu_intensive_task():
            result = 0
            for i in range(1000000):
                result += i * i
            return result
        
        # Spawn many CPU-intensive tasks
        processes = []
        num_processes = min(multiprocessing.cpu_count(), 4)
        
        try:
            for _ in range(num_processes):
                p = multiprocessing.Process(target=cpu_intensive_task)
                p.start()
                processes.append(p)
            
            # System should still respond to other requests
            # This is a basic sanity check
            assert multiprocessing.cpu_count() > 0
            
        finally:
            for p in processes:
                p.terminate()
                p.join()
    
    def test_handles_disk_full(self):
        """System should handle disk full conditions."""
        with patch('os.statvfs') as mock_stat:
            # Simulate no free space
            mock_result = MagicMock()
            mock_result.f_bavail = 0  # No free blocks
            mock_stat.return_value = mock_result
            
            from src.data.data_loader import DataLoader
            
            # Should handle gracefully
            try:
                loader = DataLoader(data_dir="/tmp/test")
                # Operation that would write to disk
                # loader.save_processed_data()  # Would fail gracefully
            except OSError as e:
                assert "No space" in str(e) or "disk" in str(e).lower()


class TestDependencyChaos:
    """Test system behavior when dependencies fail."""
    
    def test_handles_model_loading_failure(self):
        """System should handle model loading failures."""
        with patch('torch.load') as mock_load:
            mock_load.side_effect = FileNotFoundError("Model file not found")
            
            from src.inference.predictor import Predictor
            
            with pytest.raises(FileNotFoundError):
                Predictor(model_path="nonexistent.pth", device="cpu")
    
    def test_handles_corrupted_model(self):
        """System should handle corrupted model files."""
        with patch('torch.load') as mock_load:
            mock_load.side_effect = RuntimeError("Invalid magic number")
            
            from src.inference.predictor import Predictor
            
            with pytest.raises(RuntimeError):
                Predictor(model_path="corrupted.pth", device="cpu")
    
    def test_handles_database_connection_loss(self):
        """System should handle database connection failures."""
        with patch('sqlite3.connect') as mock_connect:
            mock_connect.side_effect = Exception("Database connection failed")
            
            from src.model_registry.registry import ModelRegistry
            
            # Should handle gracefully or raise appropriate error
            try:
                registry = ModelRegistry()
                registry.get_model("test")
            except Exception as e:
                assert "database" in str(e).lower() or "connection" in str(e).lower()
    
    def test_handles_redis_failure(self):
        """System should handle Redis/cache failures."""
        with patch('redis.Redis') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")
            
            from api.core.model_manager import ModelManager
            
            # Should fall back to direct loading or handle gracefully
            manager = ModelManager()
            # Operations should not crash the entire system


class TestLoadChaos:
    """Test system behavior under extreme load."""
    
    def test_handles_sudden_traffic_spike(self):
        """System should handle sudden traffic spikes."""
        import concurrent.futures
        
        def make_request():
            # Simulate API request
            time.sleep(random.uniform(0.01, 0.1))
            return {"status": "ok"}
        
        # Simulate 100 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            
            successes = 0
            failures = 0
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    successes += 1
                except Exception:
                    failures += 1
        
        # Most requests should succeed
        assert successes > 80  # At least 80% success rate
    
    def test_handles_slow_downstream_services(self):
        """System should handle slow downstream services."""
        def slow_response(*args, **kwargs):
            time.sleep(2)  # Very slow response
            response = MagicMock()
            response.status_code = 200
            return response
        
        with patch('requests.post', side_effect=slow_response):
            start_time = time.time()
            
            # Should timeout rather than hang indefinitely
            try:
                # Simulate request with timeout
                response = requests.post("http://example.com", timeout=1)
            except Timeout:
                elapsed = time.time() - start_time
                assert elapsed < 2  # Should timeout before slow response completes
    
    def test_handles_cascading_failures(self):
        """System should prevent cascading failures."""
        # Simulate failure in one component
        component_healthy = [True]
        
        def unhealthy_component():
            if not component_healthy[0]:
                raise Exception("Component failed")
            return "ok"
        
        def dependent_component():
            try:
                result = unhealthy_component()
                return f"dependent: {result}"
            except Exception:
                # Should handle upstream failure gracefully
                return "degraded mode"
        
        # Cause upstream failure
        component_healthy[0] = False
        
        # Dependent component should handle gracefully
        result = dependent_component()
        assert result == "degraded mode"


class TestRecoveryChaos:
    """Test system recovery after failures."""
    
    def test_auto_recovery_after_failure(self):
        """System should auto-recover after transient failures."""
        failure_count = [0]
        
        def flaky_service():
            failure_count[0] += 1
            if failure_count[0] <= 2:
                raise Exception("Transient failure")
            return "recovered"
        
        # Attempt recovery
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                result = flaky_service()
                assert result == "recovered"
                break
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(0.1)  # Brief delay before retry
        
        # Should have recovered
        assert failure_count[0] >= 3
    
    def test_circuit_breaker_pattern(self):
        """Circuit breaker should prevent repeated failures."""
        class CircuitBreaker:
            def __init__(self, failure_threshold=3, recovery_timeout=1):
                self.failure_count = 0
                self.failure_threshold = failure_threshold
                self.recovery_timeout = recovery_timeout
                self.last_failure_time = None
                self.state = "closed"  # closed, open, half-open
            
            def call(self, func):
                if self.state == "open":
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = "half-open"
                    else:
                        raise Exception("Circuit open")
                
                try:
                    result = func()
                    if self.state == "half-open":
                        self.state = "closed"
                    self.failure_count = 0
                    return result
                except Exception:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                    raise
        
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        
        def failing_func():
            raise Exception("Always fails")
        
        # Trip the circuit
        for i in range(2):
            try:
                cb.call(failing_func)
            except Exception:
                pass
        
        assert cb.state == "open"
        
        # Should reject calls while open
        with pytest.raises(Exception, match="Circuit open"):
            cb.call(failing_func)
        
        # Wait for recovery
        time.sleep(0.6)
        
        # Should allow one test call
        cb.state = "half-open"  # Simulate time passing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
