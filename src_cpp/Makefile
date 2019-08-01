GLIB = glib-core
SNAP = snap-core
GLIBADV = glib-adv
SNAPADV = snap-adv

SNAPROOT = ../../snap
SOURCEROOT = node2vec_ms

EXGLIB = ${SNAPROOT}/$(GLIB)
EXSNAP = ${SNAPROOT}/$(SNAP)
EXGLIBADV = ${SNAPROOT}/$(GLIBADV)
EXSNAPADV = ${SNAPROOT}/$(SNAPADV)

DEPH = $(SOURCEROOT)/n2v.h $(EXSNAPADV)/word2vec.h $(SOURCEROOT)/biasedrandomwalk.h
DEPCPP = $(SOURCEROOT)/n2v.cpp $(EXSNAPADV)/word2vec.cpp $(SOURCEROOT)/biasedrandomwalk.cpp
CXXFLAGS += $(CXXOPENMP)

MAIN = node2vec
MAINSOURCE = ${SOURCEROOT}/${MAIN}

all: ${MAINSOURCE}

# COMPILE
$(MAINSOURCE): $(MAINSOURCE).cpp $(EXSNAP)/Snap.o
	g++ $(CXXFLAGS) -o $(MAIN) $(MAINSOURCE).cpp $(DEPCPP) $(EXSNAP)/Snap.o -I$(EXSNAP) -I$(EXSNAPADV) -I$(EXGLIB) $(LDFLAGS) $(LIBS)

$(EXSNAP)/Snap.o:
	make -C $(EXSNAP)

clean:
	rm -f *.o  $(MAIN)  $(MAIN).exe
	rm -rf Debug Release
