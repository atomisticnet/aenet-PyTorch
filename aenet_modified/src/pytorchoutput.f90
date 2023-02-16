module pytorchoutput

  implicit none
  private
  save

  public :: pyo_write_init,               &
            pyo_write_final,              &
            pyo_write_header_info,        &
            pyo_write_structure_info,     &
            pyo_write_atom_sf_info,       &
            pyo_select_force_structures!, &

contains

  !--------------------------------------------------------------------!
  !                  Initialization and Finalization                   !
  !--------------------------------------------------------------------!

  subroutine pyo_write_init(pyo_unit, filename)

      implicit none
  
      character(len=*), intent(in) :: filename
      integer, intent(in)          :: pyo_unit

      open(unit = pyo_unit, action = "write", status = "replace", file = filename, form = "unformatted")    
      
  end subroutine pyo_write_init

  !--------------------------------------------------------------------!

  subroutine pyo_write_final(pyo_unit, max_nnb_trainset)

      implicit none
  
      integer, intent(in)          :: pyo_unit, max_nnb_trainset
      
      write(pyo_unit) max_nnb_trainset
      close(unit = pyo_unit) 
      
  end subroutine pyo_write_final

  !--------------------------------------------------------------------!
  !                Write Fingerprint derivatives info                  !
  !--------------------------------------------------------------------!

  subroutine pyo_write_header_info(pyo_unit, nstrucs)

      implicit none
  
      integer, intent(in)          :: pyo_unit, nstrucs

      write(pyo_unit) nstrucs
      
  end subroutine pyo_write_header_info

  !--------------------------------------------------------------------!
  !--------------------------------------------------------------------!

  subroutine pyo_write_structure_info(pyo_unit, filename, natoms, ntypes, pyo_forces_struc)

      implicit none
  
      integer, intent(in)          :: pyo_unit, natoms, ntypes, pyo_forces_struc
      character(len=*), intent(in) :: filename

      write(pyo_unit) len_trim(filename)
      write(pyo_unit) trim(filename)
      write(pyo_unit) natoms, ntypes
      write(pyo_unit) pyo_forces_struc
      
  end subroutine pyo_write_structure_info

  !--------------------------------------------------------------------!

  subroutine pyo_write_atom_sf_info(pyo_unit, itype, nnb, nsf, nblist, sfderiv_i, sfderiv_j)

      implicit none

      integer,                            intent(in) :: pyo_unit, nnb, nsf, itype
      integer,          dimension(:),     intent(in) :: nblist
      double precision, dimension(:,:),   intent(in) :: sfderiv_i
      double precision, dimension(:,:,:), intent(in) :: sfderiv_j

      double precision                               :: sfderiv_j_aux(nnb, nsf, 3), sfderiv_i_aux(nsf,3)

      integer :: ineigh, isf, icoo

      sfderiv_j_aux = 0.0d0
      sfderiv_i_aux = 0.0d0
      do ineigh = 1, nnb
        do isf = 1, nsf
          sfderiv_j_aux(ineigh, isf, :) = sfderiv_j(:, isf, ineigh)
        enddo
      enddo
      do isf = 1, nsf
        sfderiv_i_aux(isf,:) = sfderiv_i(:,isf)
      enddo

      write(pyo_unit) itype
      write(pyo_unit) nsf, nnb
      write(pyo_unit) nblist(1:nnb)
      write(pyo_unit) sfderiv_i_aux(1:nsf,1:3)
      write(pyo_unit) sfderiv_j_aux(1:nnb,1:nsf,1:3)
      
      
  end subroutine pyo_write_atom_sf_info

  !--------------------------------------------------------------------!
  !                Write Fingerprint derivatives info                  !
  !--------------------------------------------------------------------!

  subroutine pyo_select_force_structures(nstrucs, pyo_forces_percent, struc_write_force)

      implicit none
  
      integer, intent(in)               :: nstrucs
      double precision, intent(in)      :: pyo_forces_percent
      integer, allocatable, intent(out) :: struc_write_force(:)

      double precision :: u
      integer          :: istruc, jstruc, aux, N_do_forces

      allocate(struc_write_force(nstrucs))

      N_do_forces = nint(nstrucs*pyo_forces_percent)
      struc_write_force = 0
      struc_write_force(1:N_do_forces) = 1

      ! Shuffle
      do istruc = nstrucs, 2, -1
        call random_number(u)
        jstruc = int(u*istruc) + 1
        aux = struc_write_force(jstruc)
        struc_write_force(jstruc) = struc_write_force(istruc)
        struc_write_force(istruc) = aux
      enddo
      
  end subroutine pyo_select_force_structures

  !--------------------------------------------------------------------!


end module pytorchoutput