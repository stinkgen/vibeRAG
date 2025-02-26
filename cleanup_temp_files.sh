#!/bin/bash

# VibeRAG Temporary Files Cleanup Script
# This script removes temporary test files and debugging scripts

echo "🧹 Starting VibeRAG temporary files cleanup..."
echo

# Files to be removed
TEMP_FILES=(
  "test_script.py"
  "test_openai.py"
  "test_pipeline.py"
  "check_milvus.py"
  "diagnose_chat.py"
  "setup_test_data.py"
  "test.txt"
  ".env"  # Redundant with .env.local
)

# Show files that will be removed
echo "📋 The following files will be removed:"
for file in "${TEMP_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "  - $file ($(du -h "$file" | cut -f1))"
  else
    echo "  - $file (not found)"
  fi
done

echo
echo "⚠️  Note: All files in tests/ directory and cache directories will be preserved"
echo

# Confirm before deletion
read -p "Are you sure you want to delete these files? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
  echo "Cleanup canceled. No files were deleted."
  exit 0
fi

# Delete the files
echo
echo "🗑️  Removing temporary files..."
for file in "${TEMP_FILES[@]}"; do
  if [ -f "$file" ]; then
    rm "$file"
    echo "  ✅ Removed $file"
  else
    echo "  ⚠️ Skipped $file (not found)"
  fi
done

echo
echo "✅ Temporary files cleanup complete!"
echo
echo "🔎 Files preserved:"
echo "  - All files in tests/ directory (proper test suite)"
echo "  - Cache directories (__pycache__/, .pytest_cache/, etc.)"
echo "  - All core application files"
echo

chmod +x cleanup_temp_files.sh 