program trainbin2ascii

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

    to_bin = .false.
    to_ascii = .true.
    
    iarg = 1
    do while(iarg <= nargs)
       call get_command_argument(iarg, value=arg)
       select case(trim(arg))
       case('--to-binary')
          to_bin = .false.
          to_ascii = .true.
       case('--to-ascii')
          to_ascii = .false.
          to_bin = .true.
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

    integer                       :: ntypestot, nstrucs, istruc
    real*8                        :: E_scale, E_shift
    logical                       :: normalized
    character(len=2), allocatable :: type_names(:)
    real*8, allocatable           :: E_atom(:)

    integer                       :: length, natoms, ntypes, iatom, itype, jtype, nsf
    real*8                        :: energy, cooCart(3), forCart(3)
    character(len=1024)           :: filename
    real*8, allocatable           :: sfval(:)

    integer                       :: natomtot, nenv, neval, nsfparam
    real*8                        :: E_avg, E_min, E_max, rc_min, rc_max
    logical                       :: has_setups
    character(len=1024)           :: description
    character(len=2)              :: atomtype
    character(len=100)            :: sftype
    character(len=2), allocatable :: envtypes(:)
    integer, allocatable          :: sf(:), sfenv(:,:)
    real*8, allocatable           :: sfparam(:,:), sfval_min(:), sfval_max(:), sfval_avg(:), sfval_cov(:)



    open(unit = 1, action = "read", status = "old", file = infile)
    open(unit = 2, action = "write", status = "replace", file = outfile, form = "unformatted")

    ! Read header
    read(1,*) ntypestot
    read(1,*) nstrucs
    allocate(type_names(ntypestot), E_atom(ntypestot))
    read(1,*) type_names(:)
    read(1,*) E_atom(:)
    read(1,*) normalized
    read(1,*) E_scale
    read(1,*) E_shift

    write(2) ntypestot
    write(2) nstrucs
    write(2) type_names(:)
    write(2) E_atom(:)
    write(2) normalized
    write(2) E_scale
    write(2) E_shift




    ! Read dataset fingerprints
    do istruc = 1, nstrucs
        read(1,*) length
        length = min(length, len(filename))
        read(1,*) filename(1:length)
        read(1,*) natoms, ntypes
        read(1,*) energy

        write(2) length
        write(2) filename(1:length)
        write(2) natoms, ntypes
        write(2) energy

        do iatom = 1, natoms
            read(1,*) itype
            read(1,*) cooCart(:)
            read(1,*) forCart(:)
            read(1,*) nsf
            allocate(sfval(nsf))
            read(1,*) sfval(1:nsf)

            write(2) itype
            write(2) cooCart(:)
            write(2) forCart(:)
            write(2) nsf
            write(2) sfval(1:nsf)

            deallocate(sfval)
        enddo

    enddo

    ! Read footer information of the fingerprint setups
    read(1,*) natomtot
    read(1,*) E_avg, E_min, E_max
    read(1,*) has_setups

    write(2) natomtot
    write(2) E_avg, E_min, E_max
    write(2) has_setups

    do jtype = 1, ntypestot
        read(1,*) itype
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



        write(2) itype
        write(2) description
        write(2) atomtype
        write(2) nenv
        write(2,"(a)") envtypes(:)
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

        deallocate(sf, sfparam, sfenv, sfval_min, sfval_max, sfval_avg, sfval_cov, envtypes)
    enddo

    close(unit = 1)
    close(unit = 2)

end subroutine ascii2bin

  !--------------------------------------------------------------------!

subroutine bin2ascii(infile, outfile)
    implicit none

    character(len=*), intent(in)  :: infile, outfile

    integer                       :: i, ntypestot, nstrucs, istruc
    real*8                        :: E_scale, E_shift
    logical                       :: normalized
    character(len=2), allocatable :: type_names(:)
    real*8, allocatable           :: E_atom(:)

    integer                       :: length, natoms, ntypes, iatom, itype, jtype, nsf
    real*8                        :: energy, cooCart(3), forCart(3)
    character(len=1024)           :: filename
    real*8, allocatable           :: sfval(:)

    integer                       :: natomtot, nenv, neval, nsfparam
    real*8                        :: E_avg, E_min, E_max, rc_min, rc_max
    logical                       :: has_setups
    character(len=1024)           :: description
    character(len=2)              :: atomtype
    character(len=100)            :: sftype
    character(len=2), allocatable :: envtypes(:)
    integer, allocatable          :: sf(:), sfenv(:,:)
    real*8, allocatable           :: sfparam(:,:), sfval_min(:), sfval_max(:), sfval_avg(:), sfval_cov(:)



    open(unit = 1, action = "read", status = "old", file = infile, form = "unformatted")
    open(unit = 2, action = "write", status = "replace", file = outfile)

    ! Read header
    read(1) ntypestot
    read(1) nstrucs
    allocate(type_names(ntypestot), E_atom(ntypestot))
    read(1) type_names(:)
    read(1) E_atom(:)
    read(1) normalized
    read(1) E_scale
    read(1) E_shift

    write(2,*) ntypestot
    write(2,*) nstrucs
    write(2,"(100a4)") type_names(:)
    write(2,*) E_atom(:)
    write(2,*) normalized
    write(2,*) E_scale
    write(2,*) E_shift

    close(unit=1)


    ! Read footer information of the fingerprint setups
    open(unit = 1, action = "read", status = "old", file = infile, form = "unformatted")

    do i = 1, 7
        read(1)
    enddo
    do istruc = 1, nstrucs
        read(1) 
        read(1) 
        read(1) natoms, ntypes
        read(1)

        do iatom = 1, natoms
            read(1) 
            read(1) 
            read(1) 
            read(1) 
            read(1) 
        enddo

    enddo


    read(1) natomtot
    read(1) E_avg, E_min, E_max
    read(1) has_setups

    write(2,*) natomtot
    write(2,*) E_avg, E_min, E_max
    write(2,*) has_setups

    do jtype = 1, ntypestot
        read(1) itype
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



        write(2,*) itype
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

        deallocate(sf, sfparam, sfenv, sfval_min, sfval_max, sfval_avg, sfval_cov, envtypes)
    enddo

    close(unit = 1)


    ! Read dataset fingerprints
    open(unit = 1, action = "read", status = "old", file = infile, form = "unformatted")

    do i = 1, 7
        read(1)
    enddo

    do istruc = 1, nstrucs
        read(1) length
        length = min(length, len(filename))
        read(1) filename(1:length)
        read(1) natoms, ntypes
        read(1) energy

        write(2,*) length
        write(2,*) filename(1:length)
        write(2,*) natoms, ntypes
        write(2,*) energy

        do iatom = 1, natoms
            read(1) itype
            read(1) cooCart(:)
            read(1) forCart(:)
            read(1) nsf
            allocate(sfval(nsf))
            read(1) sfval(1:nsf)

            write(2,*) itype
            write(2,*) cooCart(:)
            write(2,*) forCart(:)
            write(2,*) nsf
            write(2,*) sfval(1:nsf)

            deallocate(sfval)
        enddo

    enddo

    close(unit = 1)



    close(unit = 2)

end subroutine bin2ascii

subroutine normalize_sfval(nsf, sfval_avg, sfval_cov, sfval)
    implicit none
    integer, intent(in)   :: nsf
    real*8, intent(in)    :: sfval_avg(:), sfval_cov(:)
    real*8, intent(inout) :: sfval(:)

    integer               :: isf
    real*8                :: shift, scale

    do isf = 1, nsf
        shift = sfval_avg(isf)
        scale = 1.0d0/sqrt(sfval_cov(isf) - shift**2)
        sfval(isf) = ( sfval(isf) -shift )*scale
    enddo

end subroutine normalize_sfval

end program trainbin2ascii