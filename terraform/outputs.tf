output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "ecr_api_url" {
  description = "ECR repository URL for the API image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_trainer_url" {
  description = "ECR repository URL for the trainer image"
  value       = aws_ecr_repository.trainer.repository_url
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.artifacts.bucket
}

output "rds_endpoint" {
  description = "RDS endpoint for MLflow backend store"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "kubeconfig_command" {
  description = "Command to update kubeconfig"
  value       = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
}
