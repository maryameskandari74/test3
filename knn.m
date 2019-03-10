%  example for fisherIris Neighbors k={3,5,7,15,25,}
load fisheriris
X=meas;       
Y=species;      
MaxK=25;
MaxRun=10;

KNNmodel=cell(MaxK,MaxRun);
cvmodel=cell(MaxK,MaxRun);
TotalResubLoss=zeros(MaxK,MaxRun);
TotalKFoldLoss=zeros(MaxK,MaxRun);

for k=1:MaxK 
    for run=1:MaxRun  
        KNNmodel{k,run}=ClassificationKNN.fit(X,Y,'NumNeighbors',k);
        cvmodel{k,run}=crossval(KNNmodel{k,run});
        TotalResubLoss(k,run)=resubLoss(KNNmodel{k,run});
        TotalKFoldLoss(k,run)=kfoldLoss(cvmodel{k,run});
    end
    if k==3 
       resubLoss3=resubLoss(KNNmodel{k,run});
       kfoldLoss3=kfoldLoss(cvmodel{k,run});
    end
    if k==5 
       resubLoss5=resubLoss(KNNmodel{k,run});
       kfoldLoss5=kfoldLoss(cvmodel{k,run});
    end
   
    if k==15 
      resubLoss15=resubLoss(KNNmodel{k,run});
      kfoldLoss15=kfoldLoss(cvmodel{k,run});
    end
    if k==25 
       resubLoss25=resubLoss(KNNmodel{k,run});
       kfoldLoss25=kfoldLoss(cvmodel{k,run});
    end
end
%Avg
ResubLoss=mean(TotalResubLoss,2);
KFoldLoss=mean(TotalKFoldLoss,2);
%% Select Best K
[BestValue,BestK]=min(KFoldLoss);
disp(['Best Value = ' num2str(BestValue) ])
disp(['Best K = ' num2str(BestK) ])
%%  Results
figure;
plot(ResubLoss,'r','LineWidth',2);
hold on;
plot(KFoldLoss,'b','LineWidth',2);
legend('Resub Loss','k-Fold Loss');
xlabel('Number of Neighbors - k');
ylabel('Loss');
