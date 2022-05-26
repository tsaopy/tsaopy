"""
This is a legacy file and will be probably deleted in coming updates.
"""

fortran_text_source = '''

!   commenting with ! works fine
!   line breaking with & may not work properly
!   subroutines can't properly call functions but can call other srins


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!           FORTRAN TO PYTHON MOD        !!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

!--------------------------------------------------------
!
subroutine deriv(x_in,a_in,b_in,c_in,f_in,result,na,nb,cn,cm,t)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm
real(8),intent(in) :: t
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(3) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8) :: x,v,sa,sb,sc,sf,f_amp,f_frq,f_phase
integer :: i,j

x=x_in(1)
v=x_in(2)
sa=0
sb=0
sc=0
f_amp = f_in(1)
f_frq = f_in(2)
f_phase = f_in(3)

do i=1,na
    sa = sa + a_in(i)*v*(abs(v))**(i-1)
end do
do i=1,nb
    sb = sb + b_in(i)*x**i
end do
do i=1,cn
    do j=1,cm
        sc = sc + c_in(i,j)*x**i*v**j
    end do
end do
sf = f_amp*sin(f_frq*t+f_phase)

result = (/v,sf-sa-sb-sc/)

return
end subroutine deriv

!--------------------------------------------------------
!
subroutine rk4(x_in,a_in,b_in,c_in,f_in,result,dt,na,nb,cn,cm,t)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm
real(8),intent(in) :: dt
real(8),intent(in) :: t
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(3) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8),dimension(2) :: k1,k2,k3,k4,xaux,kaux

xaux = x_in
call deriv(xaux,a_in,b_in,c_in,f_in,kaux,na,nb,cn,cm,t)
k1 = dt*kaux

xaux = x_in+0.5*k1
call deriv(xaux,a_in,b_in,c_in,f_in,kaux,na,nb,cn,cm,t+0.5*dt)
k2 = dt*kaux

xaux = x_in+0.5*k2
call deriv(xaux,a_in,b_in,c_in,f_in,kaux,na,nb,cn,cm,t+0.5*dt)
k3 = dt*kaux

xaux = x_in+k3
call deriv(xaux,a_in,b_in,c_in,f_in,kaux,na,nb,cn,cm,t+dt)
k4 = dt*kaux

result = x_in + (k1+2*k2+2*k3+k4)/6

return
end subroutine rk4

!--------------------------------------------------------
!
subroutine simulation(x_in,a_in,b_in,c_in,f_in,result,dt,nsimu,na,nb,cn,cm)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm,nsimu
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(3) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(nsimu) :: result
! aux vars
integer :: i
real(8) :: t
real(8),dimension(2) :: xaux0,xauxf

t = 0
result(1) = x_in(1)
xaux0 = x_in
do i=2,nsimu
    call rk4(xaux0,a_in,b_in,c_in,f_in,xauxf,dt,na,nb,cn,cm,t)
    result(i) = xauxf(1)
    xaux0 = xauxf
    t = t + dt
end do

return
end subroutine simulation


!--------------------------------------------------------
!
subroutine simulationv(x_in,a_in,b_in,c_in,f_in,result,dt,nsimu,na,nb,cn,cm)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm,nsimu
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(3) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(nsimu,2) :: result
! aux vars
integer :: i
real(8) :: t
real(8),dimension(2) :: xaux0,xauxf

t = 0
result(1,1) = x_in(1)
result(1,2) = x_in(2)
xaux0 = x_in
do i=2,nsimu
    call rk4(xaux0,a_in,b_in,c_in,f_in,xauxf,dt,na,nb,cn,cm,t)
    result(i,1) = xauxf(1)
    result(i,2) = xauxf(2)
    xaux0 = xauxf
    t = t + dt
end do

return
end subroutine simulationv


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

'''
import numpy.f2py
numpy.f2py.compile(fortran_text_source, modulename='f2pyauxmod',
                   verbose=False, extension='.f90')
