#!/usr/bin/env python3
"""
Schema Comparison Tool

This script compares the generated schema files with the existing ones
to help developers understand what has changed.
"""

import re
from pathlib import Path
from typing import Set, Dict, List


def extract_enum_values(file_path: Path) -> Set[str]:
    """Extract enum values from a GraphQL enum file."""
    if not file_path.exists():
        return set()

    content = file_path.read_text()
    enum_values = set()

    # Find enum blocks
    enum_pattern = r"enum\s+\w+\s*\{([^}]+)\}"
    for match in re.finditer(enum_pattern, content, re.DOTALL):
        enum_content = match.group(1)

        # Parse enum content more carefully
        lines = enum_content.split("\n")
        in_doc_comment = False

        for line in lines:
            line = line.strip()

            # Track if we're inside a documentation comment
            if line.startswith('"""'):
                in_doc_comment = not in_doc_comment
                continue

            # Skip lines inside documentation comments or regular comments
            if in_doc_comment or line.startswith("#") or not line:
                continue

            # This should be an enum value - validate it looks like one
            if re.match(r"^[A-Z][A-Z0-9_]*$", line):
                enum_values.add(line)

    return enum_values


def extract_type_fields(file_path: Path) -> Set[str]:
    """Extract field names from GraphQL type definitions."""
    if not file_path.exists():
        return set()

    content = file_path.read_text()
    fields = set()

    # Find type blocks
    type_pattern = r"type\s+\w+\s*\{([^}]+)\}"
    for match in re.finditer(type_pattern, content, re.DOTALL):
        type_content = match.group(1)
        # Extract field names
        for line in type_content.split("\n"):
            line = line.strip()
            if line and not line.startswith('"""') and not line.startswith("#"):
                # Extract field name (before colon)
                if ":" in line:
                    field_name = line.split(":")[0].strip()
                    if field_name:
                        fields.add(field_name)

    return fields


def extract_input_fields(file_path: Path) -> Set[str]:
    """Extract field names from GraphQL input definitions."""
    if not file_path.exists():
        return set()

    content = file_path.read_text()
    fields = set()

    # Find input blocks
    input_pattern = r"input\s+\w+\s*\{([^}]+)\}"
    for match in re.finditer(input_pattern, content, re.DOTALL):
        input_content = match.group(1)
        # Extract field names
        for line in input_content.split("\n"):
            line = line.strip()
            if line and not line.startswith('"""') and not line.startswith("#"):
                # Extract field name (before colon)
                if ":" in line:
                    field_name = line.split(":")[0].strip()
                    if field_name:
                        fields.add(field_name)

    return fields


def compare_sets(old_set: Set[str], new_set: Set[str], name: str) -> None:
    """Compare two sets and print the differences."""
    added = new_set - old_set
    removed = old_set - new_set

    print(f"\n{name} Comparison:")
    print("=" * 40)

    if added:
        print(f"\n✅ ADDED ({len(added)}):")
        for item in sorted(added):
            print(f"  + {item}")

    if removed:
        print(f"\n❌ REMOVED ({len(removed)}):")
        for item in sorted(removed):
            print(f"  - {item}")

    if not added and not removed:
        print("  No changes detected")

    print(f"\nSummary: {len(old_set)} → {len(new_set)} ({len(added):+d})")


def main():
    """Main comparison function."""
    print("🔍 Comparing Generated Schema with Existing Schema")
    print("=" * 60)

    # Define file paths
    current_dir = Path(".")
    parent_dir = current_dir.parent

    files_to_compare = [
        ("../llm_enums.graphql", "generated_llm_enums.graphql", extract_enum_values),
        ("../llm_inputs.graphql", "generated_llm_inputs.graphql", extract_input_fields),
        (
            "../llm_outputs.graphql",
            "generated_llm_outputs.graphql",
            extract_type_fields,
        ),
    ]

    for existing_file, generated_file, extractor_func in files_to_compare:
        existing_path = parent_dir / existing_file
        generated_path = current_dir / generated_file

        print(f"\n📄 Analyzing {existing_file}...")

        if not existing_path.exists():
            print(f"  ⚠️  Existing file not found: {existing_path}")
            continue

        if not generated_path.exists():
            print(f"  ⚠️  Generated file not found: {generated_path}")
            continue

        existing_items = extractor_func(existing_path)
        generated_items = extractor_func(generated_path)

        compare_sets(existing_items, generated_items, existing_file)

    print("\n" + "=" * 60)
    print("🎯 Comparison completed!")
    print("\n💡 Next steps:")
    print("1. Review the changes above")
    print("2. Test the generated schema in development")
    print("3. Update your schema files when ready")


if __name__ == "__main__":
    main()
