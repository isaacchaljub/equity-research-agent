# Pin Terraform + provider versions so `terraform init` is reproducible across
# machines and over time -- same idea as uv.lock pinning Python deps.
terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # No `backend` block -> Terraform defaults to local state, a
  # terraform.tfstate file written to this directory (gitignored). A remote backend 
  # becomes necessary once more than one person/machine runs `apply` against the same infra.
}

provider "aws" {
  region = var.aws_region
}
