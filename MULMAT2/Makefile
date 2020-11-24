PROJECT := mulmat2

$(PROJECT).bin: $(PROJECT).cpp $(PROJECT).cl
	g++ -Wall -O3 -std=c++0x -o $(PROJECT).bin $(PROJECT).cpp -lOpenCL

run: $(PROJECT).bin
	./$(PROJECT).bin

clean:
	rm -rf *.bin
