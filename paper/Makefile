arlrl:
	pandoc -t latex -o arlrl.pdf arlrl.md

arlrl_loop:
	while inotifywait -e close_write arlrl.md; do make aural2; done


aural2:
	pandoc -t latex -o aural2.pdf aural2.md

aural2_loop:
	while inotifywait -e close_write aural2.md; do make aural2; done
