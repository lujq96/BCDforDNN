function [Xthiss,Xthis] = UpdateX(xlast,xthiss,xthis,xnext,Wthis,Wnext,bthis,bnext,alpha,gammathis,gammanext,omega,act)
    Xthiss = xthis;
    xhat = xthis + omega * (xthis - xthiss);
    Xthistry = Wthis*xlast+bthis;
    Xthistry1 = (Xthistry>=0);
    Xnexttry = Wnext*xhat+bnext;
    Xnexttry1 = (Xnexttry>=0);
    alpha = 2/(2*gammathis + 2*gammanext*norm(Wnext,2)^2);
    %p = 0
    %alpha = 1/1000000;
    Xthis = xhat - 2*alpha*gammathis*(xhat-Xthistry1.*Xthistry) - 2*alpha*gammanext*Wnext'*(Xnexttry1.*(Xnexttry-xnext));
end