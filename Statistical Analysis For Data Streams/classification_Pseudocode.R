d=read.table("disease.txt",sep=",",header=TRUE)
attach(d)
head(d)
y=Heart.disease
n=length(y)
a=sample(n,size=n*.7)
train=(d[a,])
ytrain=train[,8]
xtrain=train[,-8]
test=(d[-a,])
xtest=test[,-8]
ytest=test[,8]

#Naive-Bayes
library(e1071)
NB=naiveBayes(xtrain,ytrain,data=train)
pred1=predict(NB,xtest)
T1=table(ActualValue=ytest,PredictedValue=pred1)
print(T1)
accuracy=(sum(diag(T1))/sum(T1))*100
sensitivity=(T1[1,1]/sum(T1[1,1],T1[1,2]))*100
specificity=(T1[2,2]/sum(T1[2,1],T1[2,2]))*100

#Decision tree
library(rpart)
library(rpart.plot)
dc=rpart(ytrain~.,data=xtrain,method="class")
rpart.plot(dc,type=1,extra=1,main="Decision Tree")
pred2=predict(dc,xtest,type="class")
T3=table(ActualValue=ytest,PredictedValue=pred2)
print(T3)
accuracy3=(sum(diag(T3))/sum(T3))*100
specificity3=(T3[1,1]/sum(T3[1,1],T3[1,2]))*100
sensitivity3=(T3[2,2]/sum(T3[2,1],T3[2,2]))*100

#Random Forest
library(randomForest)
RF=randomForest(as.factor(ytrain)~.,data=xtrain,ntree=200,importance=TRUE,proximity=TRUE);RF
pred3=predict(RF,xtest);pred3
T4=table(ActualValue=ytest,PredictedValue=pred3)
accuracy4=(sum(diag(T4))/sum(T4))*100
specificity4=(T4[1,1]/sum(T4[1,1],T4[1,2]))*100
sensitivity4=(T4[2,2]/sum(T4[2,1],T4[2,2]))*100

#Neural Network
library(nnet)
nn=nnet(class.ind(ytrain)~.,data = xtrain, size =3, maxit = 1000,softmax=TRUE)
pred4=predict(nn,xtest,type="class");pred4
T5=table(ActualValue=ytest,PredictedValue=pred4)
accuracy5=(sum(diag(T5))/sum(T5))*100
specificity5=(T5[1,1]/sum(T5[1,1],T5[1,2]))*100
sensitivity5=(T5[2,2]/sum(T5[2,1],T5[2,2]))*100

#SVM
library(e1071)
sv=svm(as.factor(ytrain)~.,data = xtrain,kernel="radial")
pred5=predict(sv,xtest);pred5
T6=table(ActualValue=ytest,PredictedValue=pred5)
accuracy6=(sum(diag(T6))/sum(T6))*100
specificity6=(T6[1,1]/sum(T6[1,1],T6[1,2]))*100
sensitivity6=(T6[2,2]/sum(T6[2,1],T6[2,2]))*100

#bagging
library(ipred)

#XGboost
library(dplyr)
library(xgboost)
library(caret)
library(archdata)
d1=as.matrix(xtrain)
d2=data.matrix(xtest)
xgboost_train = xgb.DMatrix(data=d1,label=as.factor(ytrain))
xgboost_test = xgb.DMatrix(data=d2,label=as.factor(ytest))
xb=xgboost(data = xgboost_train,max.depth = 3,nrounds=100)  
xg=xgboost(data=xgboost_train,nrounds=50,eta=0.3)
pred6=predict(xb,xgboost_test)
pred7= as.factor((ytest)[round(pred6)])
T7=table(ActualValue=ytest,PredictedValue=pred7)
accuracy7=(sum(diag(T7))/sum(T7))*100
specificity7=(T7[1,1]/sum(T7[1,1],T7[1,2]))*100
sensitivity7=(T7[2,2]/sum(T7[2,1],T7[2,2]))*100   



                        



method=c("Naive Bayes","Decision Tree","Random Forest","ANN","SVM")
acc=c(accuracy,accuracy3,accuracy4,accuracy5,accuracy6)
sens=c(sensitivity,sensitivity3,sensitivity4,sensitivity5,sensitivity6)
spec=c(specificity,specificity3,specificity4,specificity5,specificity6)
df=data.frame("Methods"=method,"Accuracy"=acc,"Sensitivity"=sens,"Specificity"=spec)
df
