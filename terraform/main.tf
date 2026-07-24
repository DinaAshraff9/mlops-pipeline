################################################################################
# MLOps Pipeline — Terraform Infrastructure
# Provisions: EKS, ECR, RDS, S3, IAM roles, VPC
################################################################################

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.30"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.13"
    }
  }

  backend "s3" {
    bucket         = "mlops-terraform-state"
    key            = "mlops-pipeline/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "mlops-pipeline"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}


################################################################################
# VPC
################################################################################
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.8"

  name = "${var.project_name}-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "production"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }
}


################################################################################
# EKS Cluster
################################################################################
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.10"

  cluster_name    = "${var.project_name}-eks"
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # OIDC provider for IRSA
  enable_irsa = true

  eks_managed_node_groups = {
    # General workloads
    general = {
      name           = "general"
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 6
      desired_size   = 2
      capacity_type  = "ON_DEMAND"

      labels = {
        role = "general"
      }
    }

    # Spot instances for training jobs (cost-optimized)
    training = {
      name           = "training"
      instance_types = ["m5.large", "m5.xlarge", "m4.xlarge"]
      min_size       = 0
      max_size       = 4
      desired_size   = 0
      capacity_type  = "SPOT"

      labels = {
        role = "training"
      }
      taints = [{
        key    = "training"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  # Add-ons
  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true }
    aws-ebs-csi-driver = { most_recent = true }
  }
}


################################################################################
# ECR Repositories
################################################################################
resource "aws_ecr_repository" "api" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_repository" "trainer" {
  name                 = "${var.project_name}-trainer"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy: keep last 10 images
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}


################################################################################
# RDS (MLflow Backend)
################################################################################
resource "aws_db_instance" "mlflow" {
  identifier           = "${var.project_name}-mlflow-db"
  engine               = "postgres"
  engine_version       = "16.2"
  instance_class       = "db.t3.micro"
  allocated_storage    = 20
  storage_encrypted    = true
  db_name              = "mlflow"
  username             = "mlflow"
  password             = var.db_password
  db_subnet_group_name = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  skip_final_snapshot  = var.environment != "production"
  deletion_protection  = var.environment == "production"
  backup_retention_period = 7
  multi_az             = var.environment == "production"

  tags = { Name = "${var.project_name}-mlflow-db" }
}

resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "${var.project_name}-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}


################################################################################
# S3 Bucket (Model Artifacts)
################################################################################
resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.project_name}-artifacts-${var.aws_account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


################################################################################
# IAM Role for API pods (IRSA)
################################################################################
module "api_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.39"

  role_name = "${var.project_name}-api-role"

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["mlops:mlops-api-sa"]
    }
  }

  role_policy_arns = {
    s3     = aws_iam_policy.s3_read.arn
    ecr    = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
    secrets = aws_iam_policy.secrets_read.arn
  }
}

resource "aws_iam_policy" "s3_read" {
  name = "${var.project_name}-s3-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_policy" "secrets_read" {
  name = "${var.project_name}-secrets-read"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = "arn:aws:secretsmanager:${var.aws_region}:${var.aws_account_id}:secret:${var.project_name}/*"
    }]
  })
}
