#!/usr/bin/env bash
# Script de build universel pour Render
set -o errexit

echo "[NLSbox Build] Installation des dépendances npm..."
npm install

echo "[NLSbox Build] Compilation du serveur TypeScript..."
npm run build

echo "[NLSbox Build] Build terminé avec succès !"
