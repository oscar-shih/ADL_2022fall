mkdir -p ckpt
wget https://www.dropbox.com/s/13pfwyreurt7hqr/intent.pt?dl=0 -O intent.pt
wget https://www.dropbox.com/s/eyq8t523dgddoki/slot.pt?dl=0 -O slot.pt
mv ./intent.pt ckpt
mv ./slot.pt ckpt