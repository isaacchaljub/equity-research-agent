#!/usr/bin/env bash
# =============================================================================
# Deploy the Equity Research Agent API to AWS ECS Fargate.
#
# Prereqs: `aws` CLI logged in (aws sts get-caller-identity works), Docker with
#          buildx, and a `.env` in the repo root holding the API keys.
# Usage:   ./deploy/aws_deploy.sh
# Override any setting inline, e.g.:  AWS_REGION=eu-west-1 ./deploy/aws_deploy.sh
#
# GROQ/GEMINI/SERPER keys are required. SENTRY_DSN, SENTRY_ENVIRONMENT and the
# LANGSMITH_* / SEC_USER_AGENT vars are picked up from .env automatically if set.
# =============================================================================
set -euo pipefail

# ---- Config (edit or override via env) --------------------------------------
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-equity-research-api}"
CLUSTER="${CLUSTER:-equity-research}"
TASK_FAMILY="${TASK_FAMILY:-equity-research}"
LOG_GROUP="${LOG_GROUP:-/ecs/equity-research}"
SECRET_NAME="${SECRET_NAME:-equity-research/keys}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
CPU="${CPU:-1024}"
MEMORY="${MEMORY:-3072}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f .env ] || { echo "ERROR: .env not found in $ROOT"; exit 1; }

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO"
echo "==> account $ACCOUNT_ID | region $AWS_REGION | $ECR_URI:$IMAGE_TAG"

# ---- A. ECR registry + build/push (amd64) -----------------------------------
echo "==> [A] ECR + build/push"
aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" \
       --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
docker buildx build --platform linux/amd64 -t "$ECR_URI:$IMAGE_TAG" --push .

# ---- B. CloudWatch log group ------------------------------------------------
echo "==> [B] log group"
aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$AWS_REGION" 2>/dev/null \
  || echo "    (log group already exists)"

# ---- C. Secrets Manager (keys read from .env, never printed) -----------------
# Writes the present keys to a Secrets Manager secret and captures their names so
# the task definition can inject exactly those (and no more).
echo "==> [C] secret"
PRESENT_KEYS="$(python3 <<'PY'
import json, pathlib
candidates = ["GROQ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY", "SENTRY_DSN", "SENTRY_ENVIRONMENT",
              "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "SEC_USER_AGENT"]
env = {}
for line in pathlib.Path(".env").read_text().splitlines():
    line = line.strip()
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
missing = [k for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "SERPER_API_KEY") if not env.get(k)]
if missing:
    raise SystemExit("ERROR: .env is missing required keys: " + ", ".join(missing))
present = {k: env[k] for k in candidates if env.get(k)}
json.dump(present, open("/tmp/equity_keys.json", "w"))
print(" ".join(present))
PY
)"
aws secretsmanager create-secret --name "$SECRET_NAME" --region "$AWS_REGION" \
  --secret-string file:///tmp/equity_keys.json >/dev/null 2>&1 \
  || aws secretsmanager put-secret-value --secret-id "$SECRET_NAME" --region "$AWS_REGION" \
       --secret-string file:///tmp/equity_keys.json >/dev/null
rm -f /tmp/equity_keys.json
SECRET_ARN="$(aws secretsmanager describe-secret --secret-id "$SECRET_NAME" --region "$AWS_REGION" --query ARN --output text)"

# ---- D. IAM execution role (pull image, write logs, read secret) ------------
echo "==> [D] IAM execution role"
cat > /tmp/ecs-trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name ecsTaskExecutionRole \
  --assume-role-policy-document file:///tmp/ecs-trust.json >/dev/null 2>&1 || echo "    (role already exists)"
aws iam attach-role-policy --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
cat > /tmp/secrets-policy.json <<EOF
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"secretsmanager:GetSecretValue","Resource":"$SECRET_ARN"}]}
EOF
aws iam put-role-policy --role-name ecsTaskExecutionRole \
  --policy-name EquityResearchSecretsRead --policy-document file:///tmp/secrets-policy.json
EXEC_ROLE_ARN="$(aws iam get-role --role-name ecsTaskExecutionRole --query 'Role.Arn' --output text)"

# ---- E. Cluster + task definition -------------------------------------------
echo "==> [E] cluster + task definition"
# Build the container "secrets" array from whichever keys were present in .env.
SECRETS_JSON=""
for k in $PRESENT_KEYS; do
  SECRETS_JSON="${SECRETS_JSON}{\"name\":\"$k\",\"valueFrom\":\"$SECRET_ARN:$k::\"},"
done
SECRETS_JSON="[${SECRETS_JSON%,}]"

aws ecs create-cluster --cluster-name "$CLUSTER" --region "$AWS_REGION" >/dev/null
cat > /tmp/taskdef.json <<EOF
{
  "family": "$TASK_FAMILY", "networkMode": "awsvpc", "requiresCompatibilities": ["FARGATE"],
  "cpu": "$CPU", "memory": "$MEMORY",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "$EXEC_ROLE_ARN",
  "containerDefinitions": [{
    "name": "equity-research", "image": "$ECR_URI:$IMAGE_TAG", "essential": true,
    "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
    "secrets": $SECRETS_JSON,
    "logConfiguration": { "logDriver": "awslogs", "options": {
      "awslogs-group": "$LOG_GROUP", "awslogs-region": "$AWS_REGION", "awslogs-stream-prefix": "ecs" }}
  }]
}
EOF
aws ecs register-task-definition --cli-input-json file:///tmp/taskdef.json --region "$AWS_REGION" >/dev/null

# ---- F. Networking + run + fetch public IP ----------------------------------
echo "==> [F] networking + run"
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$AWS_REGION")"
SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID --query 'Subnets[].SubnetId' --output text --region "$AWS_REGION" | tr '\t' ',')"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values=equity-research-sg --query 'SecurityGroups[0].GroupId' --output text --region "$AWS_REGION" 2>/dev/null || true)"
if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
  SG_ID="$(aws ec2 create-security-group --group-name equity-research-sg \
    --description 'Equity Research Agent inbound 8000' --vpc-id "$VPC_ID" --region "$AWS_REGION" --query GroupId --output text)"
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$AWS_REGION" >/dev/null
fi

aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE --task-definition "$TASK_FAMILY" --count 1 \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNETS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
  --region "$AWS_REGION" >/dev/null

TASK_ARN="$(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns[0]' --output text --region "$AWS_REGION")"
echo "    waiting for task to reach RUNNING..."
aws ecs wait tasks-running --cluster "$CLUSTER" --tasks "$TASK_ARN" --region "$AWS_REGION"
ENI="$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$TASK_ARN" \
  --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text --region "$AWS_REGION")"
PUBLIC_IP="$(aws ec2 describe-network-interfaces --network-interface-ids "$ENI" \
  --query 'NetworkInterfaces[0].Association.PublicIp' --output text --region "$AWS_REGION")"

echo ""
echo "✅ Deployed to ECS Fargate. Give it ~1-2 min to load the model, then:"
echo "   curl http://$PUBLIC_IP:8000/health"
echo "   curl -X POST http://$PUBLIC_IP:8000/query -H 'Content-Type: application/json' \\"
echo "        -d '{\"query\":\"How does AAPL current P/E compare to its 52-week range?\"}'"
echo ""
echo "Tear down with: ./deploy/aws_teardown.sh"
