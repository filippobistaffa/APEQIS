.PHONY: all

ifndef OUT
OUT=./apeqis
endif

CMP=g++ -std=c++11
WARN=-Wall -Wno-unused-result -Wno-deprecated-declarations -Wno-sign-compare -Wno-maybe-uninitialized
OPTIM=-Ofast -march=native -funroll-loops -funsafe-loop-optimizations -falign-functions=16 -falign-loops=16 -fopenmp
NOOPTIM=-O0 -march=native -fopenmp
DBG=-g ${NOOPTIM}
CUOPT=--use_fast_math -m64 -D_FORCE_INLINES -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52

INC=
LDIR=
LINK=-lcublas -lcusparse -lgomp

CUOBJSUBDIR=cuobj
COBJSUBDIR=cobj
DEPSUBDIR=dep

ECHOCC=>&2 echo "[\033[01;33m CC \033[0m]"
ECHOLD=>&2 echo "[\033[01;36m LD \033[0m]"
ECHONVCC=>&2 echo "[\033[01;32mNVCC\033[0m]"

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

define compilecuda
${ECHONVCC} $(notdir $<) ;\
mkdir -p ${DEPSUBDIR} ;\
tmp=`mktemp` ;\
${CMP} ${INC} -MM -D __CUDACC__ ${OPT} $(basename $<).h >> $$tmp ;\
if [ $$? -eq 0 ] ;\
then echo -n "${CUOBJSUBDIR}/" > ${DEPSUBDIR}/$(notdir $<).d ;\
cat $$tmp >> ${DEPSUBDIR}/$(notdir $<).d ;\
rm $$tmp ;\
mkdir -p ${CUOBJSUBDIR} ;\
cd ${CUOBJSUBDIR} ;\
nvcc -std=c++11 -c ${INC} ${CUOPT} ../$< ;\
else \
ret=$$? ;\
rm $$tmp ;\
exit $$ret ;\
fi
endef

all: apeqis
	@true

-include ${DEPSUBDIR}/*.d

apeqis: ${COBJSUBDIR}/apeqis.o ${COBJSUBDIR}/sp.o ${COBJSUBDIR}/value.o ${COBJSUBDIR}/random.o ${COBJSUBDIR}/apelib.o \
	${COBJSUBDIR}/ip.o ${CUOBJSUBDIR}/cgls.o
	@${ECHOLD} apeqis
	@nvcc ${CUOPT} ${LDIR} $^ ${LINK} -o ${OUT}

${COBJSUBDIR}/apelib.o: apelib.cpp
	@$(compilec)

${COBJSUBDIR}/value.o: value.cpp
	@$(compilec)

${COBJSUBDIR}/sp.o: sp.cpp
	@$(compilec)

${COBJSUBDIR}/random.o: random.c
	@$(compilec)

${COBJSUBDIR}/apeqis.o: apeqis.cpp
	@$(compilec)

${CUOBJSUBDIR}/cgls.o: cgls.cu
	@$(compilecuda)

${COBJSUBDIR}/ip.o: ip.cpp
	@$(compilec)

clean:
	@echo "Removing subdirectories..."
	@rm -rf ${COBJSUBDIR} ${CUOBJSUBDIR} ${DEPSUBDIR}
