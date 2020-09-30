for f in zh-CN/clips/*.mp3; do
      echo $f
      ffmpeg -loglevel panic -y -i "$f" -ar 16000 -ac 1 "${f}-16k.wav"; 
  done