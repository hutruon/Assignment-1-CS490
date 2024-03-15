download_files() {
    urls=(
        "https://www.encodeproject.org/files/ENCFF538PLU/@@download/ENCFF538PLU.bed.gz"
        "https://www.encodeproject.org/files/ENCFF298ANC/@@download/ENCFF298ANC.bed.gz"
        "https://www.encodeproject.org/files/ENCFF825KFE/@@download/ENCFF825KFE.bed.gz"
        "https://www.encodeproject.org/files/ENCFF768XXW/@@download/ENCFF768XXW.bed.gz"
        "https://www.encodeproject.org/files/ENCFF986UZO/@@download/ENCFF986UZO.bed.gz"
        "https://www.encodeproject.org/files/ENCFF092DWH/@@download/ENCFF092DWH.bed.gz"
        "https://www.encodeproject.org/files/ENCFF139HDN/@@download/ENCFF139HDN.bed.gz"
        "https://www.encodeproject.org/files/ENCFF931AKV/@@download/ENCFF931AKV.bed.gz"
        "https://www.encodeproject.org/files/ENCFF397KNY/@@download/ENCFF397KNY.bed.gz"
        "https://www.encodeproject.org/files/ENCFF615ZGF/@@download/ENCFF615ZGF.bed.gz"
    )

    for url in "${urls[@]}"; do
        wget "$url"
    done
}

# Call the function to start downloading the files
download_files
