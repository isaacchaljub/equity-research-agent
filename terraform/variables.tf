variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = <<-EOT
    Prefix for all resource names. Deliberately distinct from the names
    deploy/aws_deploy.sh uses ("equity-research-api", "ecsTaskExecutionRole",
    ...) so this Terraform-managed stack can't collide with anything the bash
    script already created in your AWS account.
  EOT
  type        = string
  default     = "equity-research-tf"
}
