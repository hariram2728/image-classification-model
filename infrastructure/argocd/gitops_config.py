"""
GitOps deployment configuration for ArgoCD.
Declarative continuous delivery for the image classification system.
"""

# ArgoCD Application manifest for staging environment
APPLICATION_STAGING = """
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: image-classifier-staging
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/image-classification.git
    targetRevision: HEAD
    path: infrastructure/helm
    helm:
      valueFiles:
        - values-staging.yaml
      parameters:
        - name: replicaCount
          value: "2"
        - name: image.tag
          value: latest
        - name: resources.limits.cpu
          value: "1"
        - name: resources.limits.memory
          value: 2Gi
  destination:
    server: https://kubernetes.default.svc
    namespace: image-classifier-staging
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
"""

# ArgoCD Application manifest for production environment
APPLICATION_PRODUCTION = """
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: image-classifier-production
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
  annotations:
    notifications.argoproj.io/subscribe.on-sync-succeeded.slack: my-channel
    notifications.argoproj.io/subscribe.on-sync-failed.slack: my-channel
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/image-classification.git
    targetRevision: v1.2.0  # Pin to specific version
    path: infrastructure/helm
    helm:
      valueFiles:
        - values-production.yaml
      parameters:
        - name: replicaCount
          value: "5"
        - name: image.pullPolicy
          value: IfNotPresent
        - name: resources.limits.cpu
          value: "2"
        - name: resources.limits.memory
          value: 4Gi
        - name: autoscaling.enabled
          value: "true"
        - name: autoscaling.minReplicas
          value: "5"
        - name: autoscaling.maxReplicas
          value: "20"
  destination:
    server: https://kubernetes.default.svc
    namespace: image-classifier-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
      - ApplyOutOfSyncOnly=true
    retry:
      limit: 10
      backoff:
        duration: 10s
        factor: 2
        maxDuration: 5m
    managedNamespaceMetadata:
      labels:
        istio-injection: enabled
"""

# ArgoCD ApplicationSet for multiple environments
APPLICATIONSET = """
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: image-classifier-appset
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - cluster: staging
            url: https://staging-k8s.example.com
            environment: staging
            replicaCount: "2"
            imageTag: latest
          - cluster: production
            url: https://production-k8s.example.com
            environment: production
            replicaCount: "5"
            imageTag: v1.2.0
  template:
    metadata:
      name: image-classifier-{{environment}}
    spec:
      project: default
      source:
        repoURL: https://github.com/your-org/image-classification.git
        targetRevision: HEAD
        path: infrastructure/helm
        helm:
          valueFiles:
            - values-{{environment}}.yaml
          parameters:
            - name: replicaCount
              value: {{replicaCount}}
            - name: image.tag
              value: {{imageTag}}
      destination:
        server: {{url}}
        namespace: image-classifier-{{environment}}
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
"""

# ArgoCD Rollout for canary deployments
ROLLOUT_CANARY = """
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: image-classifier-rollout
  namespace: image-classifier-production
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - setWeight: 25
        - pause: {duration: 10m}
        - setWeight: 50
        - pause: {duration: 15m}
        - setWeight: 75
        - pause: {duration: 10m}
        - setWeight: 100
      analysis:
        templates:
          - templateName: success-rate
          - templateName: latency-p99
        startingStep: 1
        args:
          - name: service-name
            value: image-classifier-api
  selector:
    matchLabels:
      app: image-classifier
  template:
    metadata:
      labels:
        app: image-classifier
    spec:
      containers:
        - name: api
          image: your-registry/image-classifier:v1.2.0
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "2"
              memory: 4Gi
            requests:
              cpu: "1"
              memory: 2Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
"""

# AnalysisTemplate for automated canary analysis
ANALYSIS_TEMPLATE = """
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: image-classifier-production
spec:
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.95
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            sum(rate(http_requests_total{service="image-classifier-api",status_code=~"2.."}[5m])) 
            / 
            sum(rate(http_requests_total{service="image-classifier-api"}[5m]))
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: latency-p99
  namespace: image-classifier-production
spec:
  metrics:
    - name: latency-p99
      interval: 1m
      successCondition: result[0] < 500
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            histogram_quantile(0.99, 
              sum(rate(http_request_duration_seconds_bucket{service="image-classifier-api"}[5m])) by (le)
            )
"""

# AppProject for organizing applications
APPPROJECT = """
apiVersion: argoproj.io/v1alpha1
kind: AppProject
metadata:
  name: image-classifier-project
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  description: Image Classification System
  sourceRepos:
    - https://github.com/your-org/image-classification.git
    - https://github.com/your-org/model-registry.git
  destinations:
    - namespace: image-classifier-staging
      server: https://kubernetes.default.svc
    - namespace: image-classifier-production
      server: https://kubernetes.default.svc
  clusterResourceWhitelist:
    - group: ''
      kind: Namespace
  namespaceResourceBlacklist:
    - group: ''
      kind: ResourceQuota
    - group: ''
      kind: LimitRange
  roles:
    - name: read-only
      description: Read-only access to staging and production
      policies:
        - p, proj:image-classifier-project:read-only, applications, get, image-classifier-project/*, allow
      groups:
        - image-classifier-readonly
    - name: admin
      description: Admin access to all environments
      policies:
        - p, proj:image-classifier-project:admin, applications, *, image-classifier-project/*, allow
      groups:
        - image-classifier-admin
  syncWindows:
    - kind: allow
      schedule: '0 9 * * 1-5'  # Weekdays 9 AM - 6 PM
      duration: 9h
      applications:
        - image-classifier-production
      manualSync: true
    - kind: deny
      schedule: '0 18 * * 1-5'  # Block deployments after hours
      duration: 15h
      applications:
        - image-classifier-production
      manualSync: false
"""

# Notification configuration
NOTIFICATION_CONFIG = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-notifications-cm
  namespace: argocd
data:
  service.slack: |
    token: $slack-token
  context: |
    argocdUrl: https://argocd.example.com
  template.app-deployed: |
    message: |
      {{if eq .serviceType "slack"}}:white_check_mark:{{end}} Application {{.app.metadata.name}} has been successfully deployed to {{.context.environment}}!
      Build: {{.app.spec.source.targetRevision}}
      Health: {{.app.status.health.status}}
      Sync: {{.app.status.sync.status}}
  template.app-sync-failed: |
    message: |
      {{if eq .serviceType "slack"}}:x:{{end}} Application {{.app.metadata.name}} failed to sync to {{.app.spec.source.targetRevision}} in {{.context.environment}}!
      Error: {{.app.status.operationState.message}}
  trigger.on-deployed: |
    - when: app.status.operationState.phase in ['Succeeded'] and app.status.health.status == 'Healthy'
      send: [app-deployed]
  trigger.on-sync-failed: |
    - when: app.status.operationState.phase in ['Error', 'Failed']
      send: [app-sync-failed]
"""

# Example usage in Python
def create_argocd_application(environment: str, version: str):
    """
    Programmatically create ArgoCD application configuration.
    
    Args:
        environment: Target environment (staging/production)
        version: Model/API version to deploy
    
    Returns:
        dict: ArgoCD application manifest
    """
    config = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Application",
        "metadata": {
            "name": f"image-classifier-{environment}",
            "namespace": "argocd",
        },
        "spec": {
            "project": "default",
            "source": {
                "repoURL": "https://github.com/your-org/image-classification.git",
                "targetRevision": version if environment == "production" else "HEAD",
                "path": "infrastructure/helm",
                "helm": {
                    "valueFiles": [f"values-{environment}.yaml"],
                    "parameters": [
                        {"name": "image.tag", "value": version},
                        {"name": "replicaCount", "value": "2" if environment == "staging" else "5"},
                    ]
                }
            },
            "destination": {
                "server": "https://kubernetes.default.svc",
                "namespace": f"image-classifier-{environment}"
            },
            "syncPolicy": {
                "automated": {
                    "prune": True,
                    "selfHeal": True
                }
            }
        }
    }
    
    return config


if __name__ == "__main__":
    import yaml
    
    # Generate and print configurations
    print("=== Staging Application ===")
    print(APPLICATION_STAGING)
    
    print("\n=== Production Application ===")
    print(APPLICATION_PRODUCTION)
    
    print("\n=== Canary Rollout ===")
    print(ROLLOUT_CANARY)
    
    # Example programmatic creation
    staging_app = create_argocd_application("staging", "v1.2.1-rc1")
    print("\n=== Generated Staging Config ===")
    print(yaml.dump(staging_app, default_flow_style=False))
