#!/bin/bash
# download fonts from the official google/fonts repo
# and extract the *.ttf files into the current folder

set -e

TARGET_DIR="./"
REPO_DIR="google_fonts"
REPO_URL="https://github.com/google/fonts.git"
ZIPS_DIR="zips"

mkdir -p "${TARGET_DIR}"
mkdir -p "${ZIPS_DIR}"

if [ -d "${REPO_DIR}/.git" ]; then
    echo "Updating existing Google Fonts repo..."
    git -C "${REPO_DIR}" pull --ff-only
else
    echo "Cloning Google Fonts repo..."
    git clone --depth=1 "${REPO_URL}" "${REPO_DIR}"
fi

# helper function
copy_fonts () {
    local fontdir="$1"
    # Zip the entire font directory
    zip -r "${ZIPS_DIR}/${fontdir}.zip" "${REPO_DIR}/ofl/${fontdir}"
    # Copy TTF files to target directory
    cp "${REPO_DIR}/ofl/${fontdir}"/*.ttf "${TARGET_DIR}/"
}

# former wget targets, rewritten
copy_fonts dancingscript
copy_fonts grandstander
copy_fonts notosansmono
copy_fonts recursive
copy_fonts robotoflex
copy_fonts robotomono
copy_fonts cantarell
copy_fonts petrona
copy_fonts caveat
copy_fonts ebgaramond
copy_fonts notoserif
copy_fonts newsreader

echo "Done. Fonts copied to ./${TARGET_DIR}"

# Remove the cloned repository
#rm -rf "${REPO_DIR}"

