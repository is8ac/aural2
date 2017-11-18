render:
	pandoc -t latex -o world_state.pdf arlrl_paper.md

loop: render
	while inotifywait -e close_write arlrl_paper.md; do pandoc -t latex -o world_state.pdf arlrl_paper.md; done
