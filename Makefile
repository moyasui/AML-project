compile_lorenz:
	g++ -std=c++11 -larmadillo lorenz.cpp -o lorenz

run_lorenz:
	./lorenz