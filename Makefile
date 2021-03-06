.PHONY: all

ifndef OUT
OUT=./apeqis
endif

CMP=g++
WARN=-Wall -Wno-unused-result -Wno-deprecated-declarations -Wno-sign-compare -Wno-maybe-uninitialized -Wno-ignored-attributes
OPTIM=-Ofast -march=native -funroll-loops -funsafe-loop-optimizations -falign-functions=16 -falign-loops=16 -fopenmp
NOOPTIM=-O0 -march=native -fopenmp
DBG=-g ${NOOPTIM}

CPLEXROOT=/opt/ibm/ILOG/CPLEX_Studio1271

INC=-DIL_STD -I${CPLEXROOT}/cplex/include -I${CPLEXROOT}/concert/include
LDIR=-L${CPLEXROOT}/concert/lib/x86-64_linux/static_pic -L${CPLEXROOT}/cplex/lib/x86-64_linux/static_pic
LINK=-lconcert -lilocplex -lcplex

CUOBJSUBDIR=cuobj
COBJSUBDIR=cobj
DEPSUBDIR=dep

ECHOCC=>&2 echo "[\033[01;33m CC \033[0m]"
ECHOLD=>&2 echo "[\033[01;36m LD \033[0m]"

OPT=${OPTIM} # Put desired optimisation level here

define compilec
${ECHOCC} $(notdir $<) ;\
mkdir -p ${DEPSUBDIR} ;\
tmp=`mktemp` ;\
${CMP} ${DEFS} ${INC} -MM ${OPT} $< >> $$tmp ;\
if [ $$? -eq 0 ] ;\
then echo -n "${COBJSUBDIR}/" > ${DEPSUBDIR}/$(notdir $<).d ;\
cat $$tmp >> ${DEPSUBDIR}/$(notdir $<).d ;\
rm $$tmp ;\
mkdir -p ${COBJSUBDIR} ;\
cd ${COBJSUBDIR} ;\
${CMP} ${DEFS} -c ${INC} ${OPT} ${WARN} ../$< ;\
else \
ret=$$? ;\
rm $$tmp ;\
exit $$ret ;\
fi
endef

all: apeqis
	@true

-include ${DEPSUBDIR}/*.d

apeqis: ${COBJSUBDIR}/apeqis.o ${COBJSUBDIR}/sp.o ${COBJSUBDIR}/value.o ${COBJSUBDIR}/constraints.o ${COBJSUBDIR}/random.o ${COBJSUBDIR}/apelib.o
	@${ECHOLD} apeqis
	@${CMP} ${OPT} ${LDIR} $^ ${LINK} -o ${OUT}

${COBJSUBDIR}/apelib.o: apelib.cpp
	@$(compilec)

${COBJSUBDIR}/constraints.o: constraints.cpp
	@$(compilec)

${COBJSUBDIR}/value.o: value.cpp
	@$(compilec)

${COBJSUBDIR}/sp.o: sp.cpp
	@$(compilec)

${COBJSUBDIR}/random.o: random.c
	@$(compilec)

${COBJSUBDIR}/apeqis.o: apeqis.cpp
	@$(compilec)

clean:
	@echo "Removing subdirectories..."
	@rm -rf ${COBJSUBDIR} ${CUOBJSUBDIR} ${DEPSUBDIR}
