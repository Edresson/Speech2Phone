for f in VCTK/wav48/*/*.wav; do
    echo "before substitute: $f";
    replaceby='wav22'
    outputfilename=${f//wav48/$replaceby};
    mkdir -p $(dirname "${outputfilename}")
    echo "after substitute: $outputfilename";
    ffmpeg -y -i "$f" -ar 22050 -ac 1 "$outputfilename"; 
done