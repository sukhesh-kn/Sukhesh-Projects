#A=read.csv("BC.csv");head(A)
#g1=glm(Classification~.,data=A,family="gaussian")
#summary(g1)

#A=read.csv("HR.csv");head(A)
#nrow(A)
#B=na.omit(A)
#head(B)
#nrow(B)
#write.csv(B,"HR2.csv")
#B=read.csv("HR2.csv")
#head(B)

AA=read.csv("OAnalysis.csv");head(AA)
library(moments)
summary(AA$city_development_index)
var(AA$training_hours)
sqrt(1912.284)
#original data
B=read.csv("HRR.csv");head(B,1)
nrow(B)
ncol(B)
D=B[,-1:-2]

g=step(glm(target~.,data=D,family="gaussian"))
summary(g)

#SMOTE
library(smotefamily)
library(ROSE)
set.seed(1234)

table(B$target)
table(B$company_size)
prop.table(table(B$target))  
over=ovun.sample(target~.,data=D,method="over")$data
nrow(over)
write.csv(over,"Analysis.csv")

table(over$target)
head(over)
class(over)
over
#write.csv(over,"OHR.csv")

under=ovun.sample(target~.,data=D,method="under")$data
head(under)
table(under$target)
#write.csv(over,"UHR.csv")



Z1=read.csv("Analysis.csv")
nrow(Z1)
Z=Z1[,c(-1,-6,-12)];head(Z,1)
c=cor(Z);c
round(c,2)
heatmap(c)



g1=step(glm(target~.,data=Z,family=gaussian))
summary(g1)
library(car)
a=vif(g1);a



#Train-test split
n=nrow(Z);n
library(caTools)
s=sample.split(Z,SplitRatio=0.8)
tr=subset(Z,s==TRUE)
ts=subset(Z,s==FALSE)

sp=sample(n,size=n*.8);head(sp)
tr=data.frame(Z[sp,]);head(tr)
nrow(tr)
trx=tr[,-11]
try=tr[,11]
ts=data.frame(Z[-sp,]);head(ts)
nrow(ts)
tsx=ts[,-11];head(tsx)
tsy=ts[,11];head(tsy)
nrow(tr)
nrow(ts)
class(trx)

#Logistics Regression
library(caret)
g2=glm(target~.,data=tr,family=gaussian)
summary(g2)
prob=predict(g2,tsx,type = "response")
prob=as.data.frame(prob);prob
p=ifelse(prob>0.495,1,0)
t=table(tsy,p);t

ggg=g2$fitted
G=roc(tr$target~ggg)
plot(G)
library(pROC)
roc_score = roc(tsy, prob)  # AUC score
plot(roc_score, main="ROC curve -- Logistic Regression ")

as.numeric(test_roc$auc)

(cm=confusionMatrix(as.factor(p),as.factor(tsy)))
(accuracy=sum(diag(t))/sum(t))
(recall=diag(t)/rowSums(t))
(precision=diag(t)/colSums(t))

library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
library(caret)    # for the confusionmatrix() function (also needs e1071 package)
library(dplyr)  
library(adabag)
library(caret)
library(e1071)
library(ipred)


#Random forest
library(randomForest)
rf=randomForest(as.factor(target)~.,data=tr,importance=TRUE)
rplot(rf)
barplot(table(v2))

v2=varImp(rf,scale=FALSE);v2
plot(v2[,-2])
plot(v2)
d2=as.data.frame(v2)
d22=as.data.frame(d2[,-2]);d22
rownames(d22)=rownames(d2)
d22
plot(d22)
varImpPlot(rf)
py=predict(rf,tsx);py
t1=table(tsy,py);t1
(cm1=confusionMatrix(as.factor(py),as.factor(tsy)))
(accuracy=sum(diag(t1))/sum(t1))
(recall=diag(t1)/rowSums(t1))
(precision=diag(t1)/colSums(t1))


#LDA
library("MASS")
ld=lda(target~.,data=tr)
summary(ld)
ldp=predict(ld,tsx)$class;ldp
t2=table(tsy,ldp);t2
(cm2=confusionMatrix(as.factor(ldp),as.factor(tsy)))
(accuracy=sum(diag(t2))/sum(t2))
(recall=diag(t2)/rowSums(t2))
(precision=diag(t2)/colSums(t2))


#Decision tree
library(caTools)
s=sample.split(Z,SplitRatio=0.8)
tr=subset(Z,s==TRUE)
ts=subset(Z,s==FALSE)
trx=tr[,-11]
try=tr[,11]
tsx=ts[,-11];head(tsx)
tsy=ts[,11];head(tsy)


library(rpart)
library(rpart.plot)
reg=rpart(formula = target~.,data=tr,method="class")
rpart.plot(reg,extra=1,type=1,main="Decision tree plot")
pg=predict(reg,tsx,type='class');pg
(cm3=confusionMatrix(as.factor(pg),as.factor(tsy)))
t3=table(tsy,pg);t3  
(accuracy=sum(diag(t3))/sum(t3))
(recall=diag(t3)/rowSums(t3))
(precision=diag(t3)/colSums(t3))



#KNN
library(class)
trx1=scale(trx)
tsx1=scale(tsx)
ky=knn(trx,tsx,try,k=3)
t4=table(tsy,ky);t4
(cm4=confusionMatrix(as.factor(ky),as.factor(tsy)))
(accuracy=sum(diag(t4))/sum(t4))
(recall=diag(t4)/rowSums(t4))
(precision=diag(t4)/colSums(t4))


#SVM
library(e1071)
sv=svm(target~.,data=tr)
class(tr$target)
summary(sv)
pq=predict(sv,tsx);pq
as.data.frame(pq)
t6=table(tsy,pq);t6
(cm5=confusionMatrix(as.factor(pq),as.factor(tsy)))
(accuracy=sum(diag(t6))/sum(t6))
(recall=diag(t6)/rowSums(t6))
(precision=diag(t6)/colSums(t6))
pq=as.numeric(pq)
library(pROC)
roc_score = roc(tsy, pq)  # AUC score
plot(roc_score, main="ROC curve -- Logistic Regression ")

length(try)
length(pq)
G=roc(tsy~pq);G
plot(G,xlab="1-Specificity",main="ROC Curve for Logistic Regression")


#ANN
library(nnet)
nn=class.ind(try)
seedsANN=nnet(trx,nn,,size=3,softmax = TRUE)
pnnt=predict(seedsANN,tsx,type="class")
tab=table(tsy,pnnt);tab
(cm6=confusionMatrix(as.factor(pnnt),as.factor(tsy)))
(accuracy=sum(diag(tab))/sum(tab))


#XGboost
library(xgboost)
xmatrix1=xgb.DMatrix(data=as.matrix(trx),label=try)
xmatrix2=xgb.DMatrix(data=as.matrix(tsx),label=tsy)
xg=xgboost(data=xmatrix1,nrounds=50,objective="multi:softmax",eta=0.3,num_class=2,max_depth=100)
xgp=predict(xg,xmatrix2)
tab1=table(tsy,xgp)
tab1
(cm7=confusionMatrix(as.factor(xgp),as.factor(tsy)))
(accuracy=sum(diag(tab1))/sum(tab1))



#bagging
bagging=randomForest(formula=as.factor(target)~.,data=tr,mtry=10)
pb=predict(bagging,tsx,type="class");pb
tbb=table(tsy,pb);tbb
(cm8=confusionMatrix(as.factor(pb1),as.factor(tsy)))
(accuracy=sum(diag(tbb))/sum(tbb))

#bagging
library(ipred)
bag=bagging(as.factor(target)~.,data=tr)
pbb=predict(bag,tsx)
tbb1=table(tsy,pbb);tbb1
(accuracy=sum(diag(tbb1))/sum(tbb1))


#boosting
library(gbm)
boosting=gbm(as.factor(target)~.,data=tr,distribution="bernoulli",n.trees=5000,interaction.depth = 4,shrinkage=0.2,verbose = F )
pbs=predict(boosting,tsx,type="response");pbs
pbs1=ifelse(pbs>0.499,1,0)
tbs=table(tsy,pbs1);tbs
(cm9=confusionMatrix(as.factor(pbs1),as.factor(tsy)))
(accuracy=sum(diag(tbs))/sum(tbs))

#adaboost
library(adabag)
tr$target=as.factor(tr$target)
ad=boosting(target~.,data=tr,boos=TRUE)
pada=predict(ad,tsx)
tada=table(tsy,pada$class);tada
(cm10=confusionMatrix(as.factor(pada),as.factor(tsy)))
(accuracy=sum(diag(tada))/sum(tada))

t1=ad$trees[[2]]
plot(t1)
text(t1,pretty=100)



library(ipred)
library(party)
tsx1=as.matrix(tsx)
bagCtrl <- bagControl(fit = ctreeBag$fit,
                      predict = ctreeBag$pred,
                      aggregate = ctreeBag$aggregate)
fit <- bag(target~., data = tr, bagControl = bagCtrl)
prb=predict(fit,newdata=tsx1,type="class");prb
prb1=ifelse(prb>0.499,1,0);prb1
(cm11=confusionMatrix(as.factor(prb1),as.factor(tsy)))
tpbag=table(tsy,prb)
tpbag
(accuracy=sum(diag(tada))/sum(tada))



#model stacking

xgp
pq
pbb
length(ldp)
length(py)
length(pbs1)
length(tsy)
predDF <- data.frame(xgp,pq,pbb,tsy)
modelStack <- train(as.factor(tsy) ~ ., data = predDF, method = "rf")
combPred <- predict(modelStack, predDF)
confusionMatrix(combPred, as.factor(tsy))$overall[1] 

tm=table(tsy,combPred)    ;tm

#model stacking 2
library(tidyverse)
library(h2o)
h2o.init()
h2o.getConnection() 
train_df_h2o<-as.h2o(tr)
test_df_h2o<-as.h2o(ts)
