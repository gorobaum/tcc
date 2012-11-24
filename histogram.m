function histogram(filename, titulo)
	pid = fopen(filename, "r", "native");
	dados = fscanf(pid, "%f");
	length(filename);
	hist(dados, 20);
	title(titulo);
	box;
	grid;
	xlabel("Tempo(ms)");
	ylabel("Quantidade de Threads");
	das = substr(filename, 1, length(filename)-4);
	oi 	= strcat(das,"_histo.jpg")
	print -djpg macaco.jpg;
