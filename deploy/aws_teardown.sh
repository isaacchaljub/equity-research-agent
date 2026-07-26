#!/usr/bin/env bash
# =============================================================================
# Tear down everything aws_deploy.sh created (stops billing). Order matters:
# stop the task first (releases the ENI), then delete the rest.
# Note: NOT `set -e` — we want to continue past already-deleted resources.
# =============================================================================
set -uo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
CLUSTER="${CLUSTER:-equity-research}"
ECR_REPO="${ECR_REPO:-equity-research-api}"
LOG_GROUP="${LOG_GROUP:-/ecs/equity-research}"
SECRET_NAME="${SECRET_NAME:-equity-research/keys}"

echo "==> stopping running tasks (this stops the Fargate meter)"
for t in $(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns[]' --output text --region "$AWS_REGION" 2>/dev/null); do
  aws ecs stop-task --cluster "$CLUSTER" --task "$t" --region "$AWS_REGION" >/dev/null 2>&1 || true
done

echo "==> deleting security group (retries while the ENI detaches)"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=equity-research-sg --query 'SecurityGroups[0].GroupId' --output text --region "$AWS_REGION" 2>/dev/null || true)"
if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
  for _ in 1 2 3 4 5 6; do
    aws ec2 delete-security-group --group-id "$SG_ID" --region "$AWS_REGION" 2>/dev/null && break
    echo "    ENI still attached, waiting 15s..."; sleep 15
  done
fi

echo "==> deleting log group, secret, ECR repo, cluster"
aws logs delete-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION" 2>/dev/null || true
aws secretsmanager delete-secret --secret-id "$SECRET_NAME" --force-delete-without-recovery --region "$AWS_REGION" 2>/dev/null || true
aws ecr delete-repository --repository-name "$ECR_REPO" --force --region "$AWS_REGION" 2>/dev/null || true
aws ecs delete-cluster --cluster "$CLUSTER" --region "$AWS_REGION" 2>/dev/null || true

echo ""
echo "✅ AWS teardown complete. (ecsTaskExecutionRole left in place — free + reused by future ECS work.)"
