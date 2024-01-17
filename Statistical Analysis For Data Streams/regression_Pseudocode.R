d=read.csv("regresion.csv",header=TRUE)
dd=d[-1]
attach(dd)
y=mpg
x=cbind(cylinders, displacement, horsepower, weight, acceleration)
n=length(y)
a=sample(n,size=n*.7)
train=(dd[a,])
ytrain=train[,1]
xtrain=train[,2:6]
test=(dd[-a,])
xtest=test[,2:6]
ytest=test[,1]
m=mean(ytest)
n1=length(ytest)

#SVR
library(e1071)
sv=svm(xtrain,ytrain,kernal="gaussian")
pred1=predict(sv,xtest)
SSR1=sum((pred1-m)^2)
e1=(ytest)-pred1
SSE1=sum(e1^2)
SST1=SSR1+SSE1
Rsq1=1-(SSE1/SST1)
library(Metrics)
RMSE1=rmse(ytest,pred1)

#KNN
library(caret)
library(stringi)
kn=knnreg(xtrain,ytrain)
pred2=predict(kn,xtest)
SSR2=sum((pred2-m)^2)
e2=ytest-pred2
SSE2=sum(e2^2)
SST2=SSR2+SSE2
Rsq2=1-(SSE2/SST2)
RMSE2=rmse(ytest,pred2)

#Regression tree
library(rpart)
library(rpart.plot)
dc=rpart(ytrain~.,data=xtrain)
rpart.plot(dc,type=1,extra=1,main="Decision Tree")
pred3=predict(dc,xtest)
SSR3=sum((pred3-m)^2)
e3=ytest-pred3
SSE3=sum(e3^2)
SST3=SSR3+SSE3
Rsq3=1-(SSE3/SST3)
RMSE3=rmse(ytest,pred3)

#Random Forest
library(randomForest)
rf <- randomForest(ytrain ~., data=xtrain,ntree=300)
pred4=predict(rf,xtest)
SSR4=sum((pred4-m)^2)
e4=ytest-pred4
SSE4=sum(e4^2)
SST4=SSR4+SSE4
Rsq4=1-(SSE4/SST4)
RMSE4=rmse(ytest,pred4)

#XGBoost
library(xgboost)
dmatrix1=xgb.DMatrix(data=as.matrix(xtrain),label=ytrain)
dmatrix2=xgb.DMatrix(data=as.matrix(xtest),label=ytest)
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.8
)
xg=xgboost(params = params, data = dmatrix1, nrounds = 1000)
pred5=predict(xg,dmatrix2)
SSR5=sum((pred5-m)^2)
e5=ytest-pred5
SSE5=sum(e5^2)
SST5=SSR5+SSE5
Rsq5=1-(SSE5/SST5)
RMSE5=rmse(ytest,pred5)

#Neural Network
library(nnet)
nnt=nnet(ytrain~.,data = xtrain, size =12, maxit = 500)
pred6=predict(nnt,xtest)
SSR6=sum((pred6-m)^2)
e6=ytest-pred6
SSE6=sum(e6^2)
SST6=SSR6+SSE6
Rsq6=1-(SSE6/SST6)
RMSE6=rmse(ytest,pred6)

#Adaboost

method=c("SVR","KNN Regression","Regression Tree","Random Forest","XGBoost")
rsq=c(Rsq1,Rsq2,Rsq3,Rsq4,Rsq5)
rm=c(RMSE1,RMSE2,RMSE3,RMSE4,RMSE5)
df=data.frame("Methods"=method,"Rsq"=rsq,"RMSE"=rm)
print(df)