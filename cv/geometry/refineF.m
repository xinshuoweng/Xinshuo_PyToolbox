%function F = refineF(F,pts1,pts2)
% Non-linear refinement of fundamental matrix using simplex
% pts1 and pts2 are Nx2 matrices, first row of pts1 being
% x coordinate from first image, second row of pts1 being 
% y coordinate from first image
function F = refineF(F,pts1,pts2)

X=[pts1(:,1) pts2(:,1)];
Y=[pts1(:,2) pts2(:,2)];
initF = F;

%do the minimization
minF = fminsearch (@svd_distance, initF, ...
                   optimset ('MaxFunEvals', 10000, ...
                             'MaxIter', 100000, ....
                             'Algorithm', 'levenberg-marquardt'), ...
                   X, Y);
                                        

F = rank2F(minF);



function d = svd_distance (F, X, Y)

F = rank2F(F);

homogPoints = [X(:,1), Y(:,1), ones(size(X,1),1)];
homogPointsp = [X(:,2), Y(:,2), ones(size(X,1),1)];

FX = F * homogPoints';
FTXp = F' * homogPointsp';

for i = 1:size(X,1),
    Fxi = FX(:,i);
    FTxpi = FTXp(:,i);
    dist(i) = (homogPointsp(i,:)*Fxi)^2 * ((1/(Fxi(1)^2 + Fxi(2)^2)) + (1/(FTxpi(1)^2 + FTxpi(2)^2)));
end;

d = sum(dist);


function F2 = rank2F (F)

[UF, WF, VF] = svd(F);

WF(3,3) = 0;
F2 = UF * WF * VF';
