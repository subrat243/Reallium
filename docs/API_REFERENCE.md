# API Reference

## Base URL

```
Production: https://api.deepfakedetection.com/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

All API requests require authentication using JWT tokens.

### Get Access Token

```http
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=yourpassword
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Use Token

```http
GET /detection/history
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## Endpoints

### Authentication

#### Register User

```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe",
  "role": "operator"
}
```

#### Setup MFA

```http
POST /auth/mfa/setup
Authorization: Bearer {token}
```

**Response:**
```json
{
  "secret": "JBSWY3DPEHPK3PXP",
  "qr_code_url": "otpauth://totp/..."
}
```

### Detection

#### Analyze Media

```http
POST /detection/analyze
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: @video.mp4
threshold: 0.5
return_explainability: true
```

**Response:**
```json
{
  "detection_id": "det_123abc",
  "authenticity_score": 0.85,
  "is_authentic": true,
  "is_deepfake": false,
  "confidence": 0.70,
  "modality_scores": {
    "audio": 0.82,
    "video": 0.88
  },
  "processing_time_ms": 1543
}
```

#### Batch Analysis

```http
POST /detection/batch
Authorization: Bearer {token}
Content-Type: multipart/form-data

files: @video1.mp4, @video2.mp4, @audio1.wav
threshold: 0.5
```

#### Get History

```http
GET /detection/history?limit=10&offset=0
Authorization: Bearer {token}
```

#### Submit Feedback

```http
POST /detection/feedback
Authorization: Bearer {token}
Content-Type: application/json

{
  "detection_id": "det_123abc",
  "is_correct": true,
  "notes": "Correctly identified synthetic voice"
}
```

### Model Management

#### List Models

```http
GET /models/?model_type=audio&active_only=true
Authorization: Bearer {token}
```

**Response:**
```json
{
  "models": [
    {
      "id": "model_123",
      "name": "audio_detector",
      "version": "1.2.0",
      "model_type": "audio",
      "size_mb": 18.5,
      "accuracy": 0.96,
      "is_active": true,
      "created_at": "2026-01-01T00:00:00Z"
    }
  ],
  "total": 1
}
```

#### Upload Model (Admin Only)

```http
POST /models/upload
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: @model.tflite
name: audio_detector
version: 1.3.0
model_type: audio
```

#### Check for Updates

```http
POST /models/edge/check-updates
Authorization: Bearer {token}
Content-Type: application/json

{
  "audio": "1.2.0",
  "video": "1.1.0"
}
```

**Response:**
```json
{
  "updates_available": true,
  "updates": [
    {
      "id": "model_124",
      "name": "audio_detector",
      "version": "1.3.0",
      "model_type": "audio",
      "size_mb": 19.2
    }
  ]
}
```

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 413 | Payload Too Large |
| 429 | Too Many Requests |
| 500 | Internal Server Error |

## Rate Limits

- **Per Minute**: 60 requests
- **Per Hour**: 1000 requests

Exceeding rate limits returns `429 Too Many Requests`.

## Webhooks

Configure webhooks to receive real-time notifications:

```http
POST /webhooks/configure
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["detection.completed", "model.updated"],
  "secret": "your_webhook_secret"
}
```

## SDKs

### Python

```python
from deepfake_client import DeepfakeClient

client = DeepfakeClient(
    api_key="your_api_key",
    base_url="https://api.deepfakedetection.com"
)

# Analyze media
result = client.detect("video.mp4", threshold=0.5)
print(f"Is deepfake: {result.is_deepfake}")
```

### JavaScript

```javascript
const DeepfakeClient = require('deepfake-client');

const client = new DeepfakeClient({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.deepfakedetection.com'
});

// Analyze media
const result = await client.detect('video.mp4', { threshold: 0.5 });
console.log(`Is deepfake: ${result.isDeepfake}`);
```

## Best Practices

1. **Always use HTTPS** in production
2. **Store tokens securely** (never in code)
3. **Implement retry logic** for transient failures
4. **Cache results** when appropriate
5. **Monitor rate limits** to avoid throttling
