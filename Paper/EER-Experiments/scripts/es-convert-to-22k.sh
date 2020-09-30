for f in es/clips/*.mp3; do
      echo $f
      ffmpeg -loglevel panic -y -i "$f" -ar 22050 -ac 1 "${f}-22k.wav"; 
  done