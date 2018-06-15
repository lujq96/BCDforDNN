function [Wthiss,Wthis] = updateW(xlast,xthis,wthiss,wthis,bthis,alpha,omega,act)
    Wthiss = wthis;
    What = wthis + omega * (wthis - wthiss);
    if mod(act-1,5)==0
        P=1.5;
    else
        P=2;
    end
    Xthistry = What*xlast+bthis;
    Xthistry1 = (Xthistry>=0);
    %alpha = 1/(2 * sum(sum(xlast*xlast'))); 
    alpha = 2/(2*norm(xlast,2)^2);%*norm(xlast,1)*norm(What*xlast,1));
    p = 0;
    %alpha=0.000001;
    Wthis = What - 2*alpha*Xthistry1.*(Xthistry-xthis)*xlast';
end