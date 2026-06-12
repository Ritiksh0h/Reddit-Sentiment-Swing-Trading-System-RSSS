#!/usr/bin/env bash
# push.sh — stage all changes, commit, and push to origin/main
# Usage: bash push.sh "your commit message"

set -e

if [ -z "$1" ]; then
  echo "Usage: bash push.sh \"commit message\""
  exit 1
fi

cd "$(dirname "$0")"

git add .
git commit -m "$1"
git push -u origin main
