variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "AWS Account ID"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mlops-pipeline"
}

variable "environment" {
  description = "Deployment environment (dev/staging/production)"
  type        = string
  default     = "production"
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "environment must be dev, staging, or production."
  }
}

variable "db_password" {
  description = "RDS PostgreSQL password for MLflow"
  type        = string
  sensitive   = true
}
