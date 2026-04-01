#!/bin/bash

# Schema Update Script
# This script generates new schema files and shows what changed

set -e  # Exit on any error

echo "🚀 Cloudability Schema Update Tool"
echo "=================================="
echo

# Step 1: Generate new schema files
echo "📝 Step 1: Generating schema files from live API..."
python generate_schema.py

echo
echo "✅ Schema generation completed!"
echo

# Step 2: Show comparison
echo "🔍 Step 2: Comparing with existing schema..."
python compare_schemas.py

echo
echo "🎉 Update process completed!"
echo
echo "📋 Next steps:"
echo "1. Review the generated files in this directory"
echo "2. Test the changes in development environment"  
echo "3. Manually copy files when ready:"
echo "   cp generated_llm_enums.graphql ../llm_enums.graphql"
echo "   cp generated_llm_inputs.graphql ../llm_inputs.graphql"
echo "   cp generated_llm_outputs.graphql ../llm_outputs.graphql"
echo "   cp generated_llm_helpers.graphql ../llm_helpers.graphql"
echo "4. Run 'stepzen build' to validate"
echo 