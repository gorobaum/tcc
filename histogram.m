function histogram(filename)
	pid = fopen(filename, "r", "native");
	dados = fscanf(pid, "%f");
	hist(dados, 20);
	das = substr(filename, 1, length(filename)-4);
	oi = strcat(das,"_histo.jpg");
	print(oi, "-djpg");

