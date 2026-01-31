# StreamProcess-Pipeline Kubernetes Deployment

This directory contains Kubernetes manifests for deploying StreamProcess-Pipeline to a Kubernetes cluster.

## Prerequisites

1. Kubernetes cluster (v1.24+) - works with minikube, kind, or cloud providers
2. kubectl configured to access your cluster
3. Container image built and pushed to registry
4. (Optional) ingress-nginx controller for external access
5. (Optional) cert-manager for TLS certificates
6. (Optional) Prometheus Operator for metrics

## Quick Start

### 1. Build and Push Image

```bash
# Build image
docker build -t streamprocess-pipeline:1.0.0 .

# For minikube, load directly
minikube image load streamprocess-pipeline:1.0.0

# For registry, tag and push
docker tag streamprocess-pipeline:1.0.0 your-registry/streamprocess-pipeline:1.0.0
docker push your-registry/streamprocess-pipeline:1.0.0
```

### 2. Update Secrets

**IMPORTANT**: Update secrets before deploying!

```bash
# Edit the secret file
vi 02-secret.yaml

# Or use kubectl create secret
kubectl create secret generic streamprocess-secret \
  --from-literal=POSTGRES_PASSWORD=$(openssl rand -base64 32) \
  --from-literal=SECRET_KEY=$(openssl rand -hex 32) \
  --from-literal=FLOWER_BASIC_AUTH="admin:$(openssl rand -base64 16)" \
  -n streamprocess
```

### 3. Deploy in Order

```bash
# Apply manifests in order
kubectl apply -f 00-namespace.yaml
kubectl apply -f 01-configmap.yaml
kubectl apply -f 02-secret.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/postgres-statefulset.yaml
kubectl apply -f k8s/chroma-deployment.yaml
kubectl apply -f 05-services.yaml
kubectl apply -f 03-deployment-api.yaml
kubectl apply -f 04-deployment-worker.yaml
kubectl apply -f k8s/flower-deployment.yaml
kubectl apply -f 06-ingress.yaml
kubectl apply -f 07-pdb.yaml
kubectl apply -f 09-network-policy.yaml
kubectl apply -f 08-servicemonitor.yaml  # Only if using Prometheus Operator
```

### 4. Deploy All at Once

```bash
# Deploy everything
kubectl apply -f k8s/

# Check deployment status
kubectl get all -n streamprocess
```

## Deployment Options

### Option 1: Direct kubectl

```bash
kubectl apply -f k8s/
```

### Option 2: Kustomize (Recommended)

Create a `kustomization.yaml`:

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: streamprocess

resources:
  - 00-namespace.yaml
  - 01-configmap.yaml
  - 02-secret.yaml
  - 03-deployment-api.yaml
  - 04-deployment-worker.yaml
  - 05-services.yaml
  - 06-ingress.yaml
  - 07-pdb.yaml
  - 08-servicemonitor.yaml
  - 09-network-policy.yaml
  - postgres-statefulset.yaml
  - redis-deployment.yaml
  - chroma-deployment.yaml
  - flower-deployment.yaml

# Image overrides
images:
  - name: streamprocess-pipeline
    newTag: 1.0.0

# Common labels
commonLabels:
  app: streamprocess-pipeline
  environment: production

# ConfigMap generator with checksum for automatic restarts
configMapGenerator:
  - name: streamprocess-config
    literals:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO

# Secret generator (don't commit actual secrets!)
secretGenerator:
  - name: streamprocess-secret
    literals:
      - POSTGRES_PASSWORD=CHANGE_ME
      - SECRET_KEY=CHANGE_ME
```

Deploy with kustomize:

```bash
# Apply
kustomize build . | kubectl apply -f -

# Or with kubectl
kubectl apply -k .
```

### Option 3: Helm Chart (Production)

For production, consider creating a Helm chart for better manageability.

## Verification

### Check Pod Status

```bash
# All pods
kubectl get pods -n streamprocess -w

# Describe a pod
kubectl describe pod -n streamprocess <pod-name>

# Pod logs
kubectl logs -n streamprocess <pod-name> -f

# Worker logs
kubectl logs -n streamprocess -l app=streamprocess-worker --tail=100 -f
```

### Check Services

```bash
kubectl get svc -n streamprocess

# Test API endpoint
kubectl port-forward -n streamprocess svc/streamprocess-api 8000:8000
curl http://localhost:8000/health
```

### Check Metrics

```bash
# Port forward metrics
kubectl port-forward -n streamprocess <api-pod> 9090:9090

# Access metrics
curl http://localhost:9090/metrics
```

### Check Flower (Celery Monitor)

```bash
# Port forward Flower
kubectl port-forward -n streamprocess svc/streamprocess-flower 5555:5555

# Access at http://localhost:5555
```

## Scaling

### Manual Scaling

```bash
# Scale API
kubectl scale deployment/streamprocess-api -n streamprocess --replicas=5

# Scale Workers
kubectl scale deployment/streamprocess-worker -n streamprocess --replicas=10
```

### Auto-scaling

HPA is configured and will automatically scale based on CPU/memory:

```bash
# Check HPA status
kubectl get hpa -n streamprocess

# Get HPA details
kubectl describe hpa -n streamprocess
```

## Monitoring

### Prometheus

If using Prometheus Operator, metrics are automatically scraped.

### Grafana Dashboards

Import the provided dashboards (if available):

1. Go to Grafana → Dashboards → Import
2. Upload dashboard JSON files
3. Select Prometheus datasource

### Logs

For comprehensive logging, consider:

- **Loki**: Grafana Loki for log aggregation
- **ELK**: Elasticsearch, Logstash, Kibana
- **Cloud Logging**: Stackdriver, CloudWatch, etc.

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod -n streamprocess <pod-name>

# Check logs
kubectl logs -n streamprocess <pod-name> --previous

# Common issues:
# - ImagePullBackOff: Image not accessible
# - CrashLoopBackOff: Check application logs
# - OOMKilled: Increase memory limits
```

### Connection Refused

```bash
# Check services
kubectl get svc -n streamprocess

# Check endpoints
kubectl get endpoints -n streamprocess

# Check network policies
kubectl get networkpolicy -n streamprocess
```

### High Memory/CPU Usage

```bash
# Check resource usage
kubectl top pods -n streamprocess

# Check resource limits
kubectl describe pod -n streamprocess <pod-name> | grep -A 5 Limits
```

## Cleanup

```bash
# Delete all resources
kubectl delete -f k8s/

# Or delete namespace
kubectl delete namespace streamprocess
```

## Upgrading

```bash
# Build new image
docker build -t streamprocess-pipeline:1.0.1 .

# Rolling update
kubectl set image deployment/streamprocess-api \
  streamprocess-pipeline=streamprocess-pipeline:1.0.1 \
  -n streamprocess

kubectl set image deployment/streamprocess-worker \
  streamprocess-pipeline=streamprocess-pipeline:1.0.1 \
  -n streamprocess

# Watch rollout
kubectl rollout status deployment/streamprocess-api -n streamprocess

# Rollback if needed
kubectl rollout undo deployment/streamprocess-api -n streamprocess
```

## Production Considerations

1. **Use Sealed Secrets** or external secret management
2. **Enable TLS** with cert-manager
3. **Configure resource quotas** appropriately
4. **Set up monitoring and alerting**
5. **Implement backup strategy** for databases
6. **Use persistent volumes** for data storage
7. **Configure pod disruption budgets** for HA
8. **Test disaster recovery procedures**

## Minikube Development

For local development with minikube:

```bash
# Start minikube with ingress
minikube start --driver=docker --cpus=4 --memory=8192 --ingress

# Enable ingress
minikube addons enable ingress

# Build and load image
eval $(minikube docker-env)
docker build -t streamprocess-pipeline:1.0.0 .

# Deploy
kubectl apply -f k8s/

# Access services
minikube tunnel  # For LoadBalancer services
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourorg/streamprocess-pipeline/issues
- Documentation: https://docs.example.com
