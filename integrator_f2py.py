fortran_text_source = '''

!   commenting with ! works fine
!   line breaking with & may not work properly
!   subroutines can't properly call functions but can call other srins


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!           FORTRAN TO PYTHON MOD        !!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!--------------------------------------------------------
!
subroutine deriv(x_in,a_in,b_in,result,na,nb)
implicit none
! inout vars
integer,intent(in) :: na,nb
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8) :: x,v,sa,sb
integer :: i

x=x_in(1)
v=x_in(2)
sa=0
sb=0

do i=1,na
    sa = sa + a_in(i)*v*(abs(v))**(i-1)
end do
do i=1,nb
    sb = sb + b_in(i)*x**i
end do

result = (/v,-sa-sb/)

return
end subroutine deriv

!--------------------------------------------------------
!
subroutine rk4(x_in,a_in,b_in,result,dt,na,nb)
implicit none
! inout vars
integer,intent(in) :: na,nb
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8),dimension(2) :: k1,k2,k3,k4,xaux,kaux

xaux = x_in
call deriv(xaux,a_in,b_in,kaux,na,nb)
k1 = dt*kaux

xaux = x_in+0.5*k1
call deriv(xaux,a_in,b_in,kaux,na,nb)
k2 = dt*kaux

xaux = x_in+0.5*k2
call deriv(xaux,a_in,b_in,kaux,na,nb)
k3 = dt*kaux

xaux = x_in+k3
call deriv(xaux,a_in,b_in,kaux,na,nb)
k4 = dt*kaux

result = x_in + (k1+2*k2+2*k3+k4)/6

return
end subroutine rk4

!--------------------------------------------------------
!
subroutine simulation(x_in,a_in,b_in,result,dt,nsimu,na,nb)
implicit none
! inout vars
integer,intent(in) :: na,nb,nsimu
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(out),dimension(nsimu) :: result
! aux vars
integer :: i
real(8),dimension(2) :: xaux0,xauxf

result(1) = x_in(1)
xaux0 = x_in
do i=2,nsimu
    call rk4(xaux0,a_in,b_in,xauxf,dt,na,nb)
    result(i) = xauxf(1)
    xaux0 = xauxf
end do

return
end subroutine simulation


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!                                                                   !!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''

import numpy.f2py
numpy.f2py.compile(fortran_text_source,modulename='oscadsf2py',verbose=False,
                   extension='.f90')
