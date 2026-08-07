output "ecr_repository_url" {
  description = "Push images here, e.g. docker push $(terraform output -raw ecr_repository_url):latest"
  value       = aws_ecr_repository.app.repository_url
}

output "ecs_execution_role_arn" {
  description = "Reference this from the ECS task definition you'll add next."
  value       = aws_iam_role.ecs_execution.arn
}
