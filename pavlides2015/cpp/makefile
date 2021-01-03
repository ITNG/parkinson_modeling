CXX = g++
CXXFLAGS = -std=c++11  -O2 -lm   # -Wall -Wextra

prog: main.o lib.o dde_solver.o
	$(CXX)  -o $@  $^ $(CXXFLAGS)

main.o : main.cpp lib.hpp dde_solver.hpp
	$(CXX) -c main.cpp  $(CXXFLAGS)

dde_solver.o : dde_solver.cpp lib.hpp dde_solver.hpp
	$(CXX) -c dde_solver.cpp $(CXXFLAGS)

lib.o: lib.cpp lib.hpp
	$(CXX) -c lib.cpp $(CXXFLAGS)

.PHONEY: clean
clean :
	rm -rf ../data/*

.PHONEY:eradicate
eradicate:clean
	rm -f prog *.o *.txt *.out