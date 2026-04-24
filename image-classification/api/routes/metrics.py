"""
Prometheus metrics endpoint for the API.
"""

from fastapi import APIRouter, Response
from api.core.config import settings

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    """
    # Basic metrics - in production, integrate with prometheus_client
    metrics = f"""# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{{endpoint="predict"}} 0
api_requests_total{{endpoint="batch_predict"}} 0

# HELP api_request_duration_seconds Request duration in seconds
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{{endpoint="predict",le="0.1"}} 0
api_request_duration_seconds_bucket{{endpoint="predict",le="0.5"}} 0
api_request_duration_seconds_bucket{{endpoint="predict",le="1.0"}} 0
api_request_duration_seconds_bucket{{endpoint="predict",le="+Inf"}} 0
api_request_duration_seconds_sum{{endpoint="predict"}} 0.0
api_request_duration_seconds_count{{endpoint="predict"}} 0

# HELP model_loaded Whether model is loaded
# TYPE model_loaded gauge
model_loaded 1

# HELP api_version API version information
# TYPE api_version gauge
api_version{{version="1.0.0",environment="{settings.environment}"}} 1
"""
    
    return Response(
        content=metrics,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
