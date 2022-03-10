subroutine wrf_debug(n,string)
  use grid, only: rank
  implicit none
  integer, intent(in) :: n
  character*256, intent(in) :: string

  write(*,*) '*** Error in M2005 Microphysics on processor ', rank
  write(*,*) string
  call task_abort()
end subroutine wrf_debug
