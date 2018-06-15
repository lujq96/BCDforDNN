function [Bthiss,Bthis] = updatebb(xlast,xthis,wthis,bthiss,bthis,alpha,omega,act)
    Bthiss = bthis;
    Bhat = bthis + omega * (bthis - bthiss);
    %size(wthis)
    %size(xlast)
    %size(Bhat)
    Xthistry = wthis*xlast+Bhat;
    Xthistry1 = (Xthistry>=0);
    alpha = 1/(2*60000);
    Bthis = Bhat - sum(2*alpha*Xthistry1.*(Xthistry-xthis),2);
end