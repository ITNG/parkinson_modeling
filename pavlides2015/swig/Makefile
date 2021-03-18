CXX=g++
CXXFLAGS  = -std=c++11 -fopenmp -O2  -fPIC -c
SWIGFLAGS = -c++ -python -shadow  

# location of the Python header files
PYTHON_VERSION = 3.8
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

TARGET = lib
OBJS= $(TARGET).o $(TARGET)_wrap.o

_$(TARGET).so: $(OBJS)
	$(CXX) -shared -fopenmp $(OBJS) -o _$(TARGET).so
 
$(TARGET)_wrap.o: $(TARGET)_wrap.cpp
	$(CXX) $(CXXFLAGS) $(TARGET)_wrap.cpp -I $(PYTHON_INCLUDE) 

$(TARGET).o: $(TARGET).cpp  $(TARGET).hpp
	$(CXX) $(CXXFLAGS) $(TARGET).cpp

$(TARGET)_wrap.cpp : $(TARGET).i 
	swig $(SWIGFLAGS) -o $(TARGET)_wrap.cpp $(TARGET).i 

.PHONEY:clean
clean :
	rm  -rf data/* 

.PHONEY:eradicate
eradicate: clean
	rm -rf *.o  
	rm -rf *.so 
	rm -rf $(TARGET)_wrap.cpp 
	rm -rf $(TARGET).py 
	rm -rf *.pyc
	rm -rf __pycache__
	rm -rf sbi-logs
	rm -rf lib.py

.PHONEY : clean eradicate




# swig -c++ -python  -shadow number.i
# g++ -O2 -fPIC -c number.cxx
# g++ -O2 -fPIC -c number_wrap.cxx -I /usr/include/python2.7 
# g++ -shared number.o number_wrap.o -o _number.so
