for f in zh-CN/clips/*.mp3; do
      echo $f
      ffmpeg -loglevel panic -y -i "$f" -ar 22050 -ac 1 "${f}-22k.wav"; 
  done