#-----------------------------------------------------------------------
#                   GNU Fortran Compiler
#-----------------------------------------------------------------------

LD       = gfortran

TOOL1  = trainbin2ASCII.x
TOOL2  = nnASCII2bin.x
TOOLS  = $(TARGET1) $(TARGET2)

.PHONY: tools clean

tools :
	$(LD) trainbin2ASCII.f90 -o trainbin2ASCII.x
	$(LD) nnASCII2bin.f90 -o nnASCII2bin.x

clean :
	rm -f *.x
