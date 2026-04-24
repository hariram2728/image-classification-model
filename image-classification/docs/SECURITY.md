# Security Guide

## Overview

This document outlines the security measures implemented in the Image Classification API and provides guidance for secure deployment.

## Security Vulnerabilities Fixed

### 1. Authentication & Authorization

**Issue**: Missing or weak authentication, insecure default credentials

**Fixes Implemented**:
- `AuthMiddleware` now properly registered in `api/main.py`
- Constant-time comparison for API keys to prevent timing attacks
- JWT secret validation (minimum 32 characters required)
- Empty API key defaults removed - must be explicitly configured
- Clear warnings logged when authentication is disabled

**Configuration**:
```bash
# Generate secure API key
openssl rand -hex 32

# Generate secure JWT secret
openssl rand -hex 32

# Set in .env
API_KEY=<your-64-char-hex-key>
JWT_SECRET_KEY=<your-64-char-hex-key>
```

### 2. File Upload Security

**Issue**: No validation of uploaded files, allowing malicious uploads

**Fixes Implemented**:
- File extension whitelist (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`)
- MIME type validation
- File size limits (default 10MB, configurable via `MAX_UPLOAD_SIZE_MB`)
- Image content validation using PIL's `verify()` method
- Dimension limits (max 10000x10000 pixels)
- Empty file detection

**Location**: `api/routes/predict.py::validate_image_file()`

### 3. Model Loading Security (Pickle Deserialization)

**Issue**: `torch.load()` allows arbitrary code execution via malicious pickle data

**Fixes Implemented**:
- `weights_only=True` parameter in `torch.load()` (PyTorch >= 1.13)
- Path sanitization to prevent path traversal attacks
- Checkpoint structure validation
- Type validation for all extracted values
- Graceful fallback for missing keys

**Location**: `api/core/model_manager.py::load_model()`

**Note**: For PyTorch < 1.13, consider:
- Using ONNX format for model serialization
- Implementing custom unpickler with restricted classes
- Running model loading in isolated containers

### 4. Path Traversal Prevention

**Issue**: Unsanitized file paths could allow access to arbitrary files

**Fixes Implemented**:
- `_sanitize_path()` method validates model paths
- Paths must be within project directory or `/models` folder
- Absolute path resolution and validation

**Location**: `api/core/model_manager.py::_sanitize_path()`

### 5. CORS Misconfiguration

**Issue**: Wildcard (`*`) origins with credentials enabled

**Fixes Implemented**:
- Default `ALLOWED_ORIGINS` is empty (blocks all cross-origin requests)
- Validation rejects wildcard origins when credentials are enabled
- Explicit origin list required for production
- Limited allowed methods and headers

**Configuration**:
```bash
# Development (if needed)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Production (example)
ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com

# NEVER use in production:
# ALLOWED_ORIGINS=*
```

### 6. Rate Limiting Improvements

**Issue**: Race conditions, memory leaks, no distributed support

**Fixes Implemented**:
- Bounded dictionary with maximum size (10,000 IPs)
- Periodic cleanup of old entries (every 5 minutes)
- Client fingerprinting using IP + User-Agent hash
- Rate limit headers in responses (`X-RateLimit-Limit`, `X-RateLimit-Remaining`)
- Retry-after information in 429 responses

**Future Enhancement**: Redis-backed rate limiting for distributed deployments

**Location**: `api/middleware.py::RateLimitMiddleware`

### 7. Information Disclosure Prevention

**Issue**: Stack traces and internal details exposed in error responses

**Fixes Implemented**:
- Production mode returns generic error messages only
- Debug information only included in development environment
- Exception types logged but not exposed to clients
- Correlation IDs for tracing without exposing internals

**Location**: `api/main.py::global_exception_handler()`

### 8. Security Headers

**Issue**: Missing security headers

**Fixes Implemented**:
All responses now include:
- `X-Frame-Options: DENY` (clickjacking prevention)
- `X-Content-Type-Options: nosniff` (MIME sniffing prevention)
- `X-XSS-Protection: 1; mode=block` (XSS protection)
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self'`
- `Permissions-Policy: geolocation=(), microphone=(), camera=()`
- `Strict-Transport-Security` (production with HTTPS only)

**Location**: `api/main.py::add_security_headers()`

### 9. Input Validation

**Issue**: Insufficient input validation

**Fixes Implemented**:
- Pydantic schema validation for all inputs
- Base64 image validation (format, size, dimensions)
- `top_k` parameter bounds checking (1-100)
- Batch size limits (max 100 images)
- Type coercion and validation for all parameters

**Location**: 
- `api/schemas/request.py`
- `api/routes/predict.py`

### 10. Logging Security

**Issue**: Sensitive data in logs, no correlation tracking

**Fixes Implemented**:
- Correlation IDs for request tracing
- No full request bodies logged
- Stack traces only in debug mode
- Sanitized error messages
- Structured JSON logging

**Location**: `api/middleware.py::LoggingMiddleware`

## Deployment Checklist

### Pre-Deployment

- [ ] Generate strong API key (`openssl rand -hex 32`)
- [ ] Generate strong JWT secret (min 32 chars)
- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Configure `ALLOWED_ORIGINS` with specific domains
- [ ] Set `MAX_UPLOAD_SIZE_MB` appropriately
- [ ] Review and test file upload validation
- [ ] Ensure model files are from trusted sources
- [ ] Test authentication is working

### Infrastructure

- [ ] Enable HTTPS/TLS termination
- [ ] Configure firewall rules (only expose necessary ports)
- [ ] Set up WAF (Web Application Firewall)
- [ ] Enable DDoS protection
- [ ] Configure network segmentation
- [ ] Use secrets manager for credentials (not environment variables in production)

### Monitoring

- [ ] Enable Prometheus metrics
- [ ] Set up alerting for:
  - High error rates
  - Rate limit violations
  - Authentication failures
  - Unusual traffic patterns
- [ ] Configure log aggregation
- [ ] Set up security monitoring dashboards

### Regular Maintenance

- [ ] Update dependencies regularly
- [ ] Review access logs weekly
- [ ] Rotate API keys and secrets quarterly
- [ ] Conduct security audits
- [ ] Test backup and recovery procedures
- [ ] Review and update CORS origins

## Security Best Practices

### API Key Management

1. **Never commit API keys to version control**
2. Use environment variables or secrets manager
3. Rotate keys regularly (quarterly recommended)
4. Use different keys for different environments
5. Revoke compromised keys immediately

### Model Security

1. Only load models from trusted sources
2. Validate model checksums before deployment
3. Keep PyTorch updated for latest security patches
4. Consider using ONNX for additional security
5. Run inference in isolated containers

### Data Protection

1. Encrypt data at rest and in transit
2. Implement proper access controls
3. Log access to sensitive data
4. Implement data retention policies
5. Anonymize logs where possible

### Network Security

1. Use private networks where possible
2. Implement network segmentation
3. Use VPN for administrative access
4. Regular security scanning
5. Penetration testing

## Incident Response

### If You Suspect a Security Breach

1. **Immediate Actions**:
   - Rotate all API keys and secrets
   - Review access logs for suspicious activity
   - Check for unauthorized model changes
   - Monitor for unusual traffic patterns

2. **Investigation**:
   - Preserve logs for forensic analysis
   - Identify affected systems and data
   - Determine breach scope and timeline
   - Document findings

3. **Recovery**:
   - Patch vulnerabilities
   - Restore from clean backups if needed
   - Implement additional monitoring
   - Update security procedures

4. **Post-Incident**:
   - Conduct post-mortem analysis
   - Update security documentation
   - Implement preventive measures
   - Train team on lessons learned

## Compliance Considerations

Depending on your use case, consider:

- **GDPR**: Data protection, right to erasure, consent management
- **HIPAA**: Healthcare data protection (if applicable)
- **SOC 2**: Security controls and auditing
- **ISO 27001**: Information security management

## Contact

For security issues or questions:
- Report vulnerabilities through responsible disclosure
- Do not disclose publicly before fixes are available
- Include detailed reproduction steps

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [PyTorch Security Best Practices](https://pytorch.org/docs/stable/notes/security.html)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
