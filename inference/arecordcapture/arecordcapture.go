package arecordcapture

import (
  "os/exec"
  "io"
  "os"
)

// arecord -D plughw:3 -r 16000 -t raw -f S16_LE -c 1 --max-file-time=10 /tmp/cmd/cmd.raw

// Start capturing audio
func Start()(reader io.Reader, err error){
  command := exec.Command("arecord", "-D", "plughw:1", "-r", "16000", "-t", "raw", "-f", "S16_LE", "-c", "1")
  reader, err = command.StdoutPipe()
  if err != nil {
    return
  }
  command.Stderr = os.Stderr
  command.Start()
  return
}
