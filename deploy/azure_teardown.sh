#!/usr/bin/env bash
# =============================================================================
# Tear down the Azure deployment by deleting the whole resource group.
#
# ⚠️  This deletes EVERYTHING in $RG (ACR, Key Vault, Container App, environment,
#     identity). Only safe because rg-equity-research is DEDICATED to this app.
#     If you ever point this at a shared group, delete resources individually.
# =============================================================================
set -uo pipefail

RG="${RG:-rg-equity-research}"
KV="${KV:-equityresearchkv}"

echo "==> deleting resource group '$RG' (ACR, Key Vault, app, env, identity)"
az group delete --name "$RG" --yes --no-wait
echo ""
echo "✅ Deletion started in the background (~2-3 min). Verify later with:"
echo "   az group show -n $RG        # eventually: (NotFound)"
echo ""
echo "Note: Key Vault keeps a 90-day soft-delete (no cost, name reserved)."
echo "      Purge fully if you need the name back:  az keyvault purge -n $KV"
