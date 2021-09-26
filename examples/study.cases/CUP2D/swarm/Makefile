SWIMMER = run-vracer-swarm
KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`
CUPFOLDER=../_deps/CUP-2D
CUBISMCFLAGS=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM
CUBISMLIBS= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

.SECONDARY:
.PHONY: all
all: swarm

swarm: $(SWIMMER)

$(SWIMMER): %: %.o swarmEnvironment.o ${CUPFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS)

%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

swarmEnvironment.o: _model/swarmEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS)

.PHONY: clean
clean:
	$(RM) $(SWIMMER) *.txt *.o *.ti *.optrpt *.h5 *.dat *.xmf
