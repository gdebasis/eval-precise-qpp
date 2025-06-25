#!/bin/bash

# Set directory to first argument or current directory if not provided
DIR="${1:-.}"

# Loop through all files ending with .sorted
for filepath in "$DIR"/*.sorted; do
    # Check if file exists (avoid literal pattern if no match)
    [ -e "$filepath" ] || continue

    # Compute new filename by removing .sorted
    newpath="${filepath%.sorted}"

    echo "Renaming: $filepath -> $newpath"
    mv "$filepath" "$newpath"
done

