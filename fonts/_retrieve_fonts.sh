#!/usr/bin/env bash
# Download selected Google Fonts directories via sparse checkout,
# zip each directory, and copy *.ttf files to the target directory.

set -euo pipefail

TARGET_DIR="./"
REPO_DIR="google_fonts"
REPO_URL="https://github.com/google/fonts.git"
ZIPS_DIR="zips"

FONTS=(
    dancingscript
    grandstander
    notosansmono
    recursive
    robotoflex
    robotomono
    cantarell
    petrona
    caveat
    ebgaramond
    notoserif
    newsreader
)

mkdir -p "${TARGET_DIR}" "${ZIPS_DIR}"

# Clean up any existing repo
rm -rf "${REPO_DIR}"

echo "Cloning Google Fonts repo with sparse checkout..."
git clone \
    --depth=1 \
    --filter=blob:none \
    --sparse \
    "${REPO_URL}" \
    "${REPO_DIR}"

cd "${REPO_DIR}"

git sparse-checkout init --cone
git sparse-checkout set $(printf "ofl/%s " "${FONTS[@]}")

cd ..

echo ""
echo "Processing fonts..."

for font in "${FONTS[@]}"; do
    FONT_DIR="${REPO_DIR}/ofl/${font}"
    
    if [[ ! -d "${FONT_DIR}" ]]; then
        echo "Warning: ${font} not found"
        continue
    fi
    
    echo "--> ${font}"
    
    # Create zip
    zip -qr "${ZIPS_DIR}/${font}.zip" "${FONT_DIR}"
    
    # Copy all TTF files recursively
    ttf_files=($(find "${FONT_DIR}" -type f -name "*.ttf"))
    ttf_count=${#ttf_files[@]}
    
    for ttf in "${ttf_files[@]}"; do
        cp "${ttf}" "${TARGET_DIR}/"
    done
    echo "  Copied ${ttf_count} TTF file(s)"
done

echo ""
echo "Done."
echo "TTF files --> ${TARGET_DIR}"
echo "ZIP files --> ${ZIPS_DIR}"

# Clean up
rm -rf "${REPO_DIR}"
