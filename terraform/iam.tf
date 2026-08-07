# ECS *task execution* role: what Fargate itself assumes to pull the image
# from ECR and write logs to CloudWatch on the task's behalf. This is NOT the
# app's own runtime permissions (that would be a separate "task role") --
# it's infra-level plumbing every Fargate task needs.
data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_execution" {
  name               = "${var.project_name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
}

# AWS-managed policy covering "pull from ECR, write to CloudWatch Logs" --
# the same one deploy/aws_deploy.sh attaches via `aws iam attach-role-policy`.
resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
