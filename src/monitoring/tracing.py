"""
Distributed tracing implementation using OpenTelemetry.
Provides end-to-end visibility across the image classification system.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.torch import TorchInstrumentor
from opentelemetry.semconv.trace import SpanAttributes
from opentelemetry.context import get_current
import time
from typing import Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_tracing(
    service_name: str = "image-classifier",
    jaeger_endpoint: Optional[str] = None,
    environment: str = "development",
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service for tracing
        jaeger_endpoint: Jaeger collector endpoint (optional)
        environment: Deployment environment
    
    Returns:
        Configured tracer instance
    """
    # Set up tracer provider
    provider = TracerProvider()
    
    # Add console exporter for development
    console_exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(console_exporter))
    
    # Add Jaeger exporter if endpoint provided
    if jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_endpoint.split(":")[0],
            agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        logger.info(f"Tracing configured with Jaeger endpoint: {jaeger_endpoint}")
    
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service_name)
    
    # Instrument frameworks
    try:
        FastAPIInstrumentor().instrument()
        RequestsInstrumentor().instrument()
        TorchInstrumentor().instrument()
        logger.info("Auto-instrumentation enabled for FastAPI, requests, and PyTorch")
    except Exception as e:
        logger.warning(f"Some instrumentation failed: {e}")
    
    return tracer


class TracingMiddleware:
    """
    Custom middleware for adding tracing to API requests.
    """
    
    def __init__(self, app, tracer: trace.Tracer):
        self.app = app
        self.tracer = tracer
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        
        with self.tracer.start_as_current_span(
            name=f"{method} {path}",
            kind=trace.SpanKind.SERVER,
        ) as span:
            # Set standard attributes
            span.set_attribute(SpanAttributes.HTTP_METHOD, method)
            span.set_attribute(SpanAttributes.HTTP_TARGET, path)
            span.set_attribute("service.environment", "production")
            
            # Add custom attributes from headers
            headers = dict(scope.get("headers", []))
            if b'x-correlation-id' in headers:
                correlation_id = headers[b'x-correlation-id'].decode()
                span.set_attribute("correlation.id", correlation_id)
            
            try:
                return await self.app(scope, receive, send)
            except Exception as e:
                span.record_exception(e)
                span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 500)
                raise


def trace_prediction(tracer: trace.Tracer, model_version: str):
    """
    Decorator for tracing prediction operations.
    
    Usage:
        @trace_prediction(tracer, model_version="v1.2.0")
        async def predict(image):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(
                name="model.predict",
                kind=trace.SpanKind.INTERNAL,
            ) as span:
                span.set_attribute("model.version", model_version)
                
                # Extract image metadata if available
                if args and hasattr(args[0], 'size'):
                    span.set_attribute("image.width", args[0].size[0])
                    span.set_attribute("image.height", args[0].size[1])
                
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record success metrics
                    inference_time = (time.time() - start_time) * 1000
                    span.set_attribute("inference.time_ms", inference_time)
                    span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 200)
                    
                    if isinstance(result, dict) and 'predictions' in result:
                        span.set_attribute(
                            "prediction.count",
                            len(result.get('predictions', []))
                        )
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 500)
                    raise
        
        return wrapper
    return decorator


class BusinessMetricsSpanProcessor:
    """
    Custom span processor that extracts business metrics from traces.
    """
    
    def __init__(self):
        self.metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_inference_time_ms': 0.0,
            'class_distribution': {},
        }
    
    def on_start(self, span, parent_context=None):
        pass
    
    def on_end(self, span):
        if span.name == "model.predict":
            self.metrics['total_predictions'] += 1
            
            status_code = span.attributes.get(SpanAttributes.HTTP_STATUS_CODE, 0)
            if status_code == 200:
                self.metrics['successful_predictions'] += 1
            else:
                self.metrics['failed_predictions'] += 1
            
            inference_time = span.attributes.get('inference.time_ms', 0)
            self.metrics['total_inference_time_ms'] += inference_time
            
            # Track class distribution
            predicted_class = span.attributes.get('prediction.top_class', 'unknown')
            if predicted_class not in self.metrics['class_distribution']:
                self.metrics['class_distribution'][predicted_class] = 0
            self.metrics['class_distribution'][predicted_class] += 1
    
    def shutdown(self):
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        avg_time = (
            self.metrics['total_inference_time_ms'] / 
            max(1, self.metrics['total_predictions'])
        )
        
        return {
            **self.metrics,
            'avg_inference_time_ms': avg_time,
            'success_rate': (
                self.metrics['successful_predictions'] / 
                max(1, self.metrics['total_predictions'])
            ),
        }


def create_trace_context(headers: Dict[str, str]) -> dict:
    """
    Create tracing context from incoming request headers.
    
    Args:
        headers: Request headers containing trace context
    
    Returns:
        Context object for trace propagation
    """
    # Extract traceparent header (W3C format)
    traceparent = headers.get('traceparent', '')
    
    if traceparent:
        # Parse and propagate context
        parts = traceparent.split('-')
        if len(parts) >= 4:
            trace_id = parts[1]
            parent_id = parts[2]
            # Context propagation handled automatically by OpenTelemetry
    
    return get_current()


# Example integration with FastAPI
def setup_fastapi_tracing(app, tracer: trace.Tracer):
    """
    Integrate tracing with FastAPI application.
    
    Args:
        app: FastAPI application instance
        tracer: OpenTelemetry tracer
    """
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware
    
    class TracingMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, tracer):
            super().__init__(app)
            self.tracer = tracer
        
        async def dispatch(self, request: Request, call_next):
            with self.tracer.start_as_current_span(
                name=f"{request.method} {request.url.path}",
                kind=trace.SpanKind.SERVER,
            ) as span:
                span.set_attribute(SpanAttributes.HTTP_METHOD, request.method)
                span.set_attribute(SpanAttributes.HTTP_URL, str(request.url))
                span.set_attribute(SpanAttributes.HTTP_SCHEME, request.url.scheme)
                
                # Add client info
                client_host = request.client.host if request.client else "unknown"
                span.set_attribute(SpanAttributes.NET_PEER_IP, client_host)
                
                start_time = time.time()
                
                try:
                    response = await call_next(request)
                    
                    duration = (time.time() - start_time) * 1000
                    span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)
                    span.set_attribute("response.time_ms", duration)
                    
                    return response
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 500)
                    raise
    
    app.add_middleware(TracingMiddleware, tracer=tracer)
    logger.info("FastAPI tracing middleware installed")


if __name__ == "__main__":
    # Example usage
    tracer = setup_tracing(
        service_name="image-classifier-api",
        jaeger_endpoint="localhost:6831",
        environment="development"
    )
    
    print("Tracing initialized successfully!")
    print(f"Tracer: {tracer}")
    
    # Test business metrics processor
    processor = BusinessMetricsSpanProcessor()
    
    # Simulate some spans
    with tracer.start_as_current_span("model.predict") as span:
        span.set_attribute("model.version", "v1.0.0")
        span.set_attribute("inference.time_ms", 45.2)
        span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, 200)
        span.set_attribute("prediction.top_class", "dog")
        processor.on_end(span)
    
    print(f"\nBusiness Metrics: {processor.get_metrics()}")
