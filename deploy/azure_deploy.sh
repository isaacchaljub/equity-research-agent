#!/usr/bin/env bash
# =============================================================================
# Deploy the Equity Research Agent API to Azure Container Apps — the "proper"
# path: a user-assigned managed identity pulls the image (AcrPull) and reads
# secrets from Key Vault. No admin passwords, no keys in your shell.
#
# Prereqs: `az` CLI logged in (az account show works), Docker with buildx, and a
#          `.env` in the repo root with the API keys.
# Usage:   ./deploy/azure_deploy.sh
#
# GROQ/GEMINI/SERPER are required; SENTRY_DSN + LANGSMITH_* + SEC_USER_AGENT are
# picked up from .env automatically if present.
# =============================================================================
set -euo pipefail

# ---- Config (edit or override via env) --------------------------------------
RG="${RG:-rg-equity-research}"
LOC="${LOC:-northeurope}"
ACR="${ACR:-equityresearchacr}"          # globally unique, 5-50 alphanumeric — change if taken
KV="${KV:-equityresearchkv}"             # Key Vault name; globally unique, 3-24 chars
MI="${MI:-equity-research-mi}"           # user-assigned managed identity
ENVN="${ENVN:-equity-research-env}"
APP="${APP:-equity-research}"
IMG="${IMG:-equity-research-api:v1}"
CPU="${CPU:-1.0}"
MEMORY="${MEMORY:-2.0Gi}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
[ -f .env ] || { echo "ERROR: .env not found in $ROOT"; exit 1; }

# ---- One-time subscription setup --------------------------------------------
echo "==> extension + provider registration"
az extension add --name containerapp --upgrade >/dev/null
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.KeyVault --wait
az provider register --namespace Microsoft.ManagedIdentity --wait

az group create -n "$RG" -l "$LOC" >/dev/null

# ---- A. ACR + build/push -----------------------------------------------------
# We build locally and push (docker buildx). `az acr build` would build in the
# cloud, but ACR Tasks is disabled on many free/sponsored subs.
echo "==> [A] ACR + build/push (amd64)"
az acr create -n "$ACR" -g "$RG" --sku Basic >/dev/null 2>&1 || echo "    (ACR already exists)"
az acr login -n "$ACR"
docker buildx build --platform linux/amd64 -t "$ACR.azurecr.io/$IMG" --push .

# ---- Managed identity (the app's "role") ------------------------------------
echo "==> managed identity"
az identity create -n "$MI" -g "$RG" -l "$LOC" >/dev/null 2>&1 || echo "    (identity already exists)"
MI_PRINCIPAL="$(az identity show -n "$MI" -g "$RG" --query principalId -o tsv)"
MI_RESID="$(az identity show -n "$MI" -g "$RG" --query id -o tsv)"

# ---- Key Vault (RBAC) + secrets from .env -----------------------------------
echo "==> Key Vault + secrets"
az keyvault create -n "$KV" -g "$RG" -l "$LOC" --enable-rbac-authorization true >/dev/null 2>&1 || echo "    (KV already exists)"
KV_ID="$(az keyvault show -n "$KV" -g "$RG" --query id -o tsv)"
ACR_ID="$(az acr show -n "$ACR" --query id -o tsv)"
ME="$(az ad signed-in-user show --query id -o tsv)"

# You (Owner) need a DATA-plane role to write secrets — Owner alone can't under RBAC.
az role assignment create --assignee "$ME" --role "Key Vault Secrets Officer" --scope "$KV_ID" >/dev/null 2>&1 || true
echo "    waiting 45s for RBAC propagation before writing secrets..."
sleep 45

# Which keys are present in .env? (validates the required three.)
PRESENT_KEYS="$(python3 <<'PY'
import pathlib
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
print(" ".join(k for k in candidates if env.get(k)))
PY
)"

# read a value from .env without exporting it into the shell environment
val() { grep "^$1=" .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'"; }

# Set one Key Vault secret per present key, and build the containerapp args.
SECRETS_ARGS=""
ENV_ARGS=""
for k in $PRESENT_KEYS; do
  kv_name="$(echo "$k" | tr '[:upper:]_' '[:lower:]-')"   # GROQ_API_KEY -> groq-api-key
  az keyvault secret set --vault-name "$KV" -n "$kv_name" --value "$(val "$k")" >/dev/null
  SECRETS_ARGS="$SECRETS_ARGS ${kv_name}=keyvaultref:https://$KV.vault.azure.net/secrets/$kv_name,identityref:$MI_RESID"
  ENV_ARGS="$ENV_ARGS ${k}=secretref:${kv_name}"
done

# grant the identity: read Key Vault secrets + pull from ACR
az role assignment create --assignee "$MI_PRINCIPAL" --role "Key Vault Secrets User" --scope "$KV_ID" >/dev/null 2>&1 || true
az role assignment create --assignee "$MI_PRINCIPAL" --role AcrPull --scope "$ACR_ID" >/dev/null 2>&1 || true

# ---- Environment + app -------------------------------------------------------
echo "==> Container Apps environment"
az containerapp env create -n "$ENVN" -g "$RG" -l "$LOC" >/dev/null

echo "==> Container App (identity + Key Vault-referenced secrets + HTTPS ingress)"
az containerapp create -n "$APP" -g "$RG" \
  --environment "$ENVN" \
  --image "$ACR.azurecr.io/$IMG" \
  --target-port 8000 --ingress external \
  --cpu "$CPU" --memory "$MEMORY" --min-replicas 1 --max-replicas 3 \
  --user-assigned "$MI_RESID" \
  --registry-server "$ACR.azurecr.io" --registry-identity "$MI_RESID" \
  --secrets $SECRETS_ARGS \
  --env-vars $ENV_ARGS \
  >/dev/null

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo ""
echo "✅ Deployed to Container Apps. Give it ~1-2 min, then (note HTTPS):"
echo "   curl https://$FQDN/health"
echo "   curl -X POST https://$FQDN/query -H 'Content-Type: application/json' \\"
echo "        -d '{\"query\":\"How does AAPL current P/E compare to its 52-week range?\"}'"
echo ""
echo "Tear down with: ./deploy/azure_teardown.sh"
