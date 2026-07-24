# 🏆 End-to-End MLOps Pipeline

> **Production-grade** MLOps platform: from model training to Kubernetes deployment with full monitoring, drift detection, and CI/CD automation.

[![CI/CD](https://github.com/your-org/mlops-pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-org/mlops-pipeline/actions)
[![codecov](https://codecov.io/gh/your-org/mlops-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/your-org/mlops-pipeline)

---

## 🏗️ Architecture

```
Code Push ──► GitHub Actions ──► Tests & Security Scan
                    │
                    ▼
             Docker Build ──► ECR Push ──► Trivy Scan
                    │
                    ▼
          Kubernetes (EKS) ──► Rolling Deploy ──► Smoke Tests
                    │
                    ▼
     MLflow Tracking ◄──── Training Job (Optuna Tuning)
                    │
                    ▼
    FastAPI Prediction API ──► Prometheus ──► Grafana
                    │
                    ▼
        Evidently Drift Detection ──► Auto Retrain
```

---

## ✨ Features

| Feature | Technology |
|---------|-----------|
| CI/CD Pipeline | GitHub Actions |
| Experiment Tracking | MLflow + PostgreSQL |
| Hyperparameter Tuning | Optuna |
| Containerization | Docker (multi-stage) |
| Container Registry | AWS ECR + Trivy scanning |
| Orchestration | Kubernetes (EKS) |
| Auto Scaling | HPA (CPU + custom metrics) |
| Zero-Downtime Deploy | Rolling Update strategy |
| Monitoring | Prometheus + Grafana |
| Alerting | Alertmanager → Slack / PagerDuty |
| Logging | ELK Stack |
| Drift Detection | Evidently AI |
| Infrastructure | Terraform (IaC) |
| Security | IRSA + Secrets Manager + Bandit |
| API | FastAPI + Rate Limiting + Auth |

---

## 🚀 Quick Start (Local)

### Prerequisites
- Docker & Docker Compose
- Python 3.11+

### 1. Clone & setup
```bash
git clone https://github.com/your-org/mlops-pipeline.git
cd mlops-pipeline
cp .env.example .env
```

### 2. Generate sample data
```bash
python scripts/generate_sample_data.py
```

### 3. Start the stack
```bash
# Start core services (MLflow, API, Prometheus, Grafana)
docker compose -f docker/docker-compose.yml up -d

# Run training job
docker compose -f docker/docker-compose.yml --profile train run trainer
```

### 4. Access services
| Service | URL |
|---------|-----|
| Prediction API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## 🔮 Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer mlops-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"features": [150.0, 14, 5, 45.2, 2, 10, 120.0, 1, 36, 720]}'
```

Response:
```json
{
  "prediction": 0,
  "probability": 0.8734,
  "model_version": "1.0.0",
  "latency_ms": 12.4
}
```

---

## 🏛️ AWS Infrastructure

```bash
cd terraform
terraform init
terraform plan -var="aws_account_id=123456789" -var="db_password=secure123"
terraform apply
```

---

## 🔄 Trigger Retraining

Add `[retrain]` to your commit message to automatically trigger the training pipeline:
```bash
git commit -m "feat: updated feature engineering [retrain]"
```

Or dispatch manually from GitHub Actions UI.

---

## 📊 Grafana Dashboard

Import `monitoring/grafana/dashboards/mlops-dashboard.json` for:
- Real-time predictions per second
- P50 / P95 / P99 latency
- Model accuracy gauge
- Error rate trending
- Active requests

---

## 🧪 Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=api --cov=model
```

---

## 📁 Project Structure

```
mlops-pipeline/
├── api/                    ← FastAPI prediction service
│   ├── main.py
│   └── requirements.txt
├── model/                  ← Training & evaluation
│   ├── train.py            ← Optuna + MLflow training
│   └── evaluate.py         ← Drift detection (Evidently)
├── docker/                 ← Docker configs
│   ├── Dockerfile.api
│   ├── Dockerfile.trainer
│   └── docker-compose.yml
├── k8s/                    ← Kubernetes manifests
│   ├── namespace.yaml
│   ├── api-deployment.yaml ← Rolling update + Ingress
│   ├── hpa.yaml            ← Auto-scaling
│   ├── mlflow-deployment.yaml
│   ├── pvc.yaml
│   └── secrets.yaml
├── terraform/              ← AWS Infrastructure (IaC)
│   ├── main.tf             ← EKS, ECR, RDS, S3, IAM
│   ├── variables.tf
│   └── outputs.tf
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   ├── alerts.yml      ← 10+ alert rules
│   │   └── alertmanager.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── github-workflows/
│   ├── ci-cd.yml           ← Full CI/CD pipeline
│   └── drift-detection.yml ← Scheduled drift checks
├── tests/
│   ├── test_api.py
│   └── test_model.py
├── scripts/
│   └── generate_sample_data.py
└── README.md
```

---

## 👤 Author

Built as a portfolio project showcasing Cloud + DevOps + MLOps expertise.

**Tech Stack:** AWS EKS · Terraform · Kubernetes · Docker · MLflow · FastAPI · Prometheus · Grafana · GitHub Actions · Python
