# ECR repository to hold the Docker image. Terraform only creates the
# *repository* -- it never builds or pushes the image itself; that stays a
# `docker push` step (see deploy/aws_deploy.sh section [A] for the imperative
# AWS CLI equivalent of everything in this file).
resource "aws_ecr_repository" "app" {
  name                 = "${var.project_name}-api"
  image_tag_mutability = "MUTABLE" # aws_deploy.sh reuses one tag ("v1") on every push; switch to IMMUTABLE once CI tags images by git SHA.

  image_scanning_configuration {
    scan_on_push = true # matches --image-scanning-configuration scanOnPush=true in the bash script
  }
}
