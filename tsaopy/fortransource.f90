!--------------------------------------------------------
!
subroutine deriv(x_in,a_in,b_in,c_in,result,na,nb,cn,cm,f_i)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm
real(8),intent(in) :: f_i
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8) :: x,v,sa,sb,sc
integer :: i,j
x=x_in(1)
v=x_in(2)
sa=0
sb=0
sc=0
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
result = (/v,f_i-sa-sb-sc/)
return
end subroutine deriv
!--------------------------------------------------------
!
subroutine rk4(x_in,a_in,b_in,c_in,result,dt,na,nb,cn,cm,f0,f1,f2)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm
real(8),intent(in) :: dt,f0,f1,f2
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(2) :: result
! aux vars
real(8),dimension(2) :: k1,k2,k3,k4,xaux,kaux
xaux = x_in
call deriv(xaux,a_in,b_in,c_in,kaux,na,nb,cn,cm,f0)
k1 = dt*kaux
xaux = x_in+0.5*k1
call deriv(xaux,a_in,b_in,c_in,kaux,na,nb,cn,cm,f1)
k2 = dt*kaux
xaux = x_in+0.5*k2
call deriv(xaux,a_in,b_in,c_in,kaux,na,nb,cn,cm,f1)
k3 = dt*kaux
xaux = x_in+k3
call deriv(xaux,a_in,b_in,c_in,kaux,na,nb,cn,cm,f2)
k4 = dt*kaux
result = x_in + (k1+2*k2+2*k3+k4)/6
return
end subroutine rk4
!--------------------------------------------------------
!
subroutine simulation(x_in,a_in,b_in,c_in,f_in,dt,datalen,na,nb,cn,cm,nf,result)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm,datalen,nf
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(nf) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(datalen) :: result
! aux vars
integer :: i
real(8) :: f0,f1,f2
real(8),dimension(2) :: xaux0,xauxf
result(1) = x_in(1)
xaux0 = x_in
do i=1,datalen-1
    f0=f_in(2*i-1)
    f1=f_in(2*i)
    f2=f_in(2*i+1)
    call rk4(xaux0,a_in,b_in,c_in,xauxf,dt,na,nb,cn,cm,f0,f1,f2)
    result(i+1) = xauxf(1)
    xaux0 = xauxf
end do
return
end subroutine simulation
!--------------------------------------------------------
!
subroutine simulationv(x_in,a_in,b_in,c_in,f_in,dt,datalen,na,nb,cn,cm,nf,result)
implicit none
! inout vars
integer,intent(in) :: na,nb,cn,cm,datalen,nf
real(8),intent(in) :: dt
real(8),intent(in),dimension(2) :: x_in
real(8),intent(in),dimension(nf) :: f_in
real(8),intent(in),dimension(na) :: a_in
real(8),intent(in),dimension(nb) :: b_in
real(8),intent(in),dimension(cn,cm) :: c_in
real(8),intent(out),dimension(datalen,2) :: result
! aux vars
integer :: i
real(8) :: f0,f1,f2
real(8),dimension(2) :: xaux0,xauxf
result(1,1) = x_in(1)
result(1,2) = x_in(2)
xaux0 = x_in
do i=1,datalen-1
    f0=f_in(2*i-1)
    f1=f_in(2*i)
    f2=f_in(2*i+1)
    call rk4(xaux0,a_in,b_in,c_in,xauxf,dt,na,nb,cn,cm,f0,f1,f2)
    result(i+1,1) = xauxf(1)
    result(i+1,2) = xauxf(2)
    xaux0 = xauxf
end do
return
end subroutine simulationv

