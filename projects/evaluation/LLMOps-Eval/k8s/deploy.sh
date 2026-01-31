#!/bin/bash
# LLMOps-Eval Kubernetes Deployment Script
#
# Usage:
#   ./deploy.sh              # Deploy all resources
#   ./deploy.sh secrets      # Create secrets only
#   ./deploy.sh delete       # Delete all resources

set -e

# Configuration
NAMESPACE="${NAMESPACE:-default}"
REGISTRY="${REGISTRY:-docker.io/your-registry}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
RELEASE_NAME="${RELEASE_NAME:-llmops-eval}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo-info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo-warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo-error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if kubectl is available
check-kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo-error "kubectl is not installed or not in PATH"
        exit 1
    fi
    echo-info "kubectl found: $(kubectl version --client --short 2>/dev/null || echo 'unknown')"
}

# Create namespace if needed
create-namespace() {
    echo-info "Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
}

# Create secrets
create-secrets() {
    echo-info "Creating secrets..."
    kubectl apply -f k8s/secret.yaml
}

# Create ConfigMaps
create-configmaps() {
    echo-info "Creating ConfigMaps..."
    kubectl apply -f k8s/configmap.yaml
}

# Create PVCs
create-pvcs() {
    echo-info "Creating PersistentVolumeClaims..."
    kubectl apply -f k8s/configmap.yaml -f k8s/pvc.yaml
}

# Deploy API
deploy-api() {
    echo-info "Deploying API..."
    kubectl apply -f k8s/deployment.yaml
    kubectl apply -f k8s/service.yaml

    # Wait for rollout
    echo-info "Waiting for API deployment to be ready..."
    kubectl rollout status deployment/llmops-eval-api -n "$NAMESPACE" --timeout=5m
}

# Deploy Ingress
deploy-ingress() {
    echo-info "Deploying Ingress..."
    kubectl apply -f k8s/ingress.yaml
}

# Get status
get-status() {
    echo-info "Deployment status:"
    echo ""
    kubectl get all -n "$NAMESPACE" -l app=llmops-eval
    echo ""
    echo-info "Services:"
    kubectl get svc -n "$NAMESPACE" -l app=llmops-eval
    echo ""
    echo-info "Ingress:"
    kubectl get ingress -n "$NAMESPACE" -l app=llmops-eval
}

# Get logs
get-logs() {
    echo-info "Fetching logs from API pods..."
    kubectl logs -n "$NAMESPACE" -l component=api --tail=50 -f
}

# Delete all resources
delete-all() {
    echo-warn "Deleting all LLMOps-Eval resources..."
    kubectl delete -f k8s/ingress.yaml --ignore-not-found=true
    kubectl delete -f k8s/service.yaml --ignore-not-found=true
    kubectl delete -f k8s/deployment.yaml --ignore-not-found=true
    kubectl delete -f k8s/configmap.yaml --ignore-not-found=true
    kubectl delete -f k8s/secret.yaml --ignore-not-found=true
    echo-info "Resources deleted"
}

# Main deployment
deploy-all() {
    check-kubectl
    create-namespace
    create-secrets
    create-configmaps
    create-pvcs
    deploy-api
    deploy-ingress
    get-status

    echo ""
    echo-info "Deployment complete!"
    echo ""
    echo "To access the API:"
    echo "  kubectl port-forward -n $NAMESPACE svc/llmops-eval-api 8002:8000"
    echo ""
    echo "To view logs:"
    echo "  ./deploy.sh logs"
}

# Parse arguments
case "${1:-deploy}" in
    deploy)
        deploy-all
        ;;
    secrets)
        check-kubectl
        create-secrets
        ;;
    status)
        check-kubectl
        get-status
        ;;
    logs)
        check-kubectl
        get-logs
        ;;
    delete)
        check-kubectl
        delete-all
        ;;
    *)
        echo "Usage: $0 {deploy|secrets|status|logs|delete}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy all resources (default)"
        echo "  secrets  - Create secrets only"
        echo "  status   - Show deployment status"
        echo "  logs     - Show logs from API pods"
        echo "  delete   - Delete all resources"
        exit 1
        ;;
esac
