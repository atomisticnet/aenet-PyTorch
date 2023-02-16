program nnASCII2bin

implicit none

character(len=1024) :: infile, outfile
logical             :: to_bin, to_ascii



call initialize(infile, outfile, to_bin, to_ascii)

if (to_ascii .and. .not. to_bin) then
    call bin2ascii(infile, outfile)
else if (to_bin .and. .not. to_ascii) then
    call ascii2bin(infile, outfile)
endif


contains

subroutine initialize(infile, outfile, to_bin, to_ascii)

    implicit none

    character(len=*), intent(out) :: infile, outfile
    integer :: iarg, nargs
    character(len=100) :: arg
    logical, intent(out) :: to_bin, to_ascii

    nargs = command_argument_count()
    if (nargs < 1) then
       write(0,*) "Error: No input file provided."
       call finalize()
       stop
    end if

    infile = ' '
    outfile = ' '

    to_bin = .true.
    to_ascii = .false.
    
    iarg = 1
    do while(iarg <= nargs)
       call get_command_argument(iarg, value=arg)
       select case(trim(arg))
       case('--to-binary')
          to_bin = .true.
          to_ascii = .false.
       case('--to-ascii')
          to_ascii = .true.
          to_bin = .false.
       case default
          if (len_trim(infile) == 0) then
             infile = trim(arg)
          else if (len_trim(outfile) == 0) then
             outfile = trim(arg)
          else
             write(0,*) 'Error: Unknown argument: ', trim(arg)
             call finalize()
             stop
          end if
       end select
       iarg = iarg + 1
    end do

    if ((len(infile) == 0) .or. (len(outfile) == 0))then
       write(0,*) 'Error: No input file specified.'
       call finalize()
       stop
    end if

end subroutine initialize

  !--------------------------------------------------------------------!

subroutine finalize()

    implicit none

end subroutine finalize

  !--------------------------------------------------------------------!

subroutine ascii2bin(infile, outfile)
    implicit none

    character(len=*), intent(in)  :: infile, outfile

    integer              :: nlayers, nnodesmax, Wsize, nvalues
    integer, allocatable :: nnodes(:), fun(:), iw(:), iv(:)
    real*8, allocatable  :: W(:)

    character(len=1024)           :: description
    character(len=100)            :: sftype
    character(len=2)              :: atomtype
    character(len=2), allocatable :: envtypes(:)
    real*8                        :: rc_min, rc_max
    integer                       :: nsf, nsfparam, neval, nenv
    integer, allocatable          :: sf(:), sfenv(:,:)
    real*8, allocatable           :: sfparam(:,:), sfval_min(:), sfval_max(:), sfval_avg(:), sfval_cov(:)

    character(len=1024)           :: file
    logical                       :: normalized
    real*8                        :: scale, shift, E_min, E_max, E_avg
    integer                       :: ntypes, natomtot, nstrucs
    character(len=2), allocatable :: type_names(:)
    real*8, allocatable           :: E_atom(:)


    open(unit = 1, action = "read", status = "old", file = infile)
    open(unit = 2, action = "write", status = "replace", file = outfile, form = "unformatted")


    ! Network information
    read(1,*) nlayers
    read(1,*) nnodesmax
    read(1,*) Wsize
    read(1,*) nvalues

    allocate(nnodes(nlayers), fun(nlayers-1), iw(nlayers), iv(nlayers), W(Wsize))

    read(1,*) nnodes(:)
    read(1,*) fun(:)
    read(1,*) iw(:)
    read(1,*) iv(:)
    read(1,*) W(:)


    write(2) nlayers
    write(2) nnodesmax
    write(2) Wsize
    write(2) nvalues
    write(2) nnodes(:)
    write(2) fun(:)
    write(2) iw(:)
    write(2) iv(:)
    write(2) W(:)


    deallocate(nnodes, fun, iw, iv, W)



    ! Structural Fingerprint setup information
    read(1,*) description
    read(1,*) atomtype
    read(1,*) nenv

    allocate(envtypes(nenv))

    read(1,*) envtypes(:)
    read(1,*) rc_min
    read(1,*) rc_max
    read(1,*) sftype
    read(1,*) nsf
    read(1,*) nsfparam

    allocate(sf(nsf), sfparam(nsfparam,nsf), sfenv(2,nsf), sfval_min(nsf), &
             sfval_max(nsf), sfval_avg(nsf), sfval_cov(nsf))

    read(1,*) sf(:)
    read(1,*) sfparam(:,:)
    read(1,*) sfenv(:,:)
    read(1,*) neval
    read(1,*) sfval_min
    read(1,*) sfval_max
    read(1,*) sfval_avg
    read(1,*) sfval_cov

    !print*, sfparam(:,1)


    write(2) description
    write(2) atomtype
    write(2) nenv
    write(2) envtypes(:)
    write(2) rc_min
    write(2) rc_max
    write(2) sftype
    write(2) nsf
    write(2) nsfparam
    write(2) sf(:)
    write(2) sfparam(:,:)
    write(2) sfenv(:,:)
    write(2) neval
    write(2) sfval_min
    write(2) sfval_max
    write(2) sfval_avg
    write(2) sfval_cov

    deallocate(sf, sfparam, sfenv, sfval_min, sfval_max, sfval_avg, sfval_cov)



    ! Trainset information
    read(1,*) file
    read(1,*) normalized
    read(1,*) scale
    read(1,*) shift
    read(1,*) ntypes

    allocate(type_names(ntypes), E_atom(ntypes))

    read(1,*) type_names(:)
    read(1,*) E_atom(:)
    read(1,*) natomtot
    read(1,*) nstrucs
    read(1,*) E_min, E_max, E_avg


    write(2) file
    write(2) normalized
    write(2) scale
    write(2) shift
    write(2) ntypes
    write(2) type_names(:)
    write(2) E_atom(:)
    write(2) natomtot
    write(2) nstrucs
    write(2) E_min, E_max, E_avg

    deallocate(type_names, E_atom)


    close(unit = 1)
    close(unit = 2)

end subroutine ascii2bin

  !--------------------------------------------------------------------!

subroutine bin2ascii(infile, outfile)
    implicit none

    character(len=*), intent(in)  :: infile, outfile

    integer              :: nlayers, nnodesmax, Wsize, nvalues
    integer, allocatable :: nnodes(:), fun(:), iw(:), iv(:)
    real*8, allocatable  :: W(:)

    character(len=1024)           :: description
    character(len=100)            :: sftype
    character(len=2)              :: atomtype
    character(len=2), allocatable :: envtypes(:)
    real*8                        :: rc_min, rc_max
    integer                       :: nsf, nsfparam, neval, nenv
    integer, allocatable          :: sf(:), sfenv(:,:)
    real*8, allocatable           :: sfparam(:,:), sfval_min(:), sfval_max(:), sfval_avg(:), sfval_cov(:)

    character(len=1024)           :: file
    logical                       :: normalized
    real*8                        :: scale, shift, E_min, E_max, E_avg
    integer                       :: ntypes, natomtot, nstrucs
    character(len=2), allocatable :: type_names(:)
    real*8, allocatable           :: E_atom(:)


    open(unit = 1, action = "read", status = "old", file = infile, form = "unformatted")
    open(unit = 2, action = "write", status = "replace", file = outfile)


    ! Network information
    read(1) nlayers
    read(1) nnodesmax
    read(1) Wsize
    read(1) nvalues

    allocate(nnodes(nlayers), fun(nlayers-1), iw(nlayers), iv(nlayers), W(Wsize))

    read(1) nnodes(:)
    read(1) fun(:)
    read(1) iw(:)
    read(1) iv(:)
    read(1) W(:)


    write(2,*) nlayers
    write(2,*) nnodesmax
    write(2,*) Wsize
    write(2,*) nvalues
    write(2,*) nnodes(:)
    write(2,*) fun(:)
    write(2,*) iw(:)
    write(2,*) iv(:)
    write(2,*) W(:)

    deallocate(nnodes, fun, iw, iv, W)



    ! Structural Fingerprint setup information
    read(1) description
    read(1) atomtype
    read(1) nenv

    allocate(envtypes(nenv))

    read(1) envtypes(:)
    read(1) rc_min
    read(1) rc_max
    read(1) sftype
    read(1) nsf
    read(1) nsfparam

    allocate(sf(nsf), sfparam(nsfparam,nsf), sfenv(2,nsf), sfval_min(nsf), &
             sfval_max(nsf), sfval_avg(nsf), sfval_cov(nsf))

    read(1) sf(:)
    read(1) sfparam(:,:)
    read(1) sfenv(:,:)
    read(1) neval
    read(1) sfval_min
    read(1) sfval_max
    read(1) sfval_avg
    read(1) sfval_cov



    write(2,*) description
    write(2,*) atomtype
    write(2,*) nenv
    write(2,"(a)") envtypes(:)
    write(2,*) rc_min
    write(2,*) rc_max
    write(2,*) sftype
    write(2,*) nsf
    write(2,*) nsfparam
    write(2,*) sf(:)
    write(2,*) sfparam(:,:)
    write(2,*) sfenv(:,:)
    write(2,*) neval
    write(2,*) sfval_min
    write(2,*) sfval_max
    write(2,*) sfval_avg
    write(2,*) sfval_cov

    deallocate(sf, sfparam, sfenv, sfval_min, sfval_max, sfval_avg, sfval_cov)



    ! Trainset information
    read(1) file
    read(1) normalized
    read(1) scale
    read(1) shift
    read(1) ntypes

    allocate(type_names(ntypes), E_atom(ntypes))

    read(1) type_names(:)
    read(1) E_atom(:)
    read(1) natomtot
    read(1) nstrucs
    read(1) E_min, E_max, E_avg


    write(2,*) file
    write(2,*) normalized
    write(2,*) scale
    write(2,*) shift
    write(2,*) ntypes
    write(2,*) type_names(:)
    write(2,*) E_atom(:)
    write(2,*) natomtot
    write(2,*) nstrucs
    write(2,*) E_min, E_max, E_avg

    deallocate(type_names, E_atom)


    close(unit = 1)
    close(unit = 2)

end subroutine bin2ascii

end program nnASCII2bin