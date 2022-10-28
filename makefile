all:
	g++ -std=c++11 -O3 -g -Wall -fmessage-length=0 -o threes threes.cpp
stats:
	./threes --total=1000 --save=stats.txt --play='load=weights.bin alpha=0'
train:
	./threes --play='save=weights.bin alpha=0.001' --total=100000 --block=1000 --limit=1000
keep_train:
	./threes --play='load=weights.bin save=weights.bin alpha=0.001' --total=100000 --block=1000 --limit=1000
judge:
	/tcgdisk/threes-judge --load stats.txt --judge version=2
clean:
	rm threes