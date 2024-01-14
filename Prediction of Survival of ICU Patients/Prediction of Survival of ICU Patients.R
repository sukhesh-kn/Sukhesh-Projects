A=read.csv("ICU.csv");head(A)
attach(A)
cancer=ifelse(A$case=='cancer',1,0)                      
kidney_injury=ifelse(A$case=='kidney injury',1,0)
stroke=ifelse(A$case=="stroke",1,0)
heart_failure=ifelse(A$case=="heart failure",1,0)
hepatic_failure=ifelse(A$case=="hepatic failure",1,0)
respiratory=ifelse(A$case=="respiratory",1,0)
brain_injury=ifelse(A$case=="brain injury",1,0)
Vital_Status=ifelse(A$vital.status=='survival',1,0)
Sex=ifelse(A$sex=="M",1,0)
x=cbind(A,Sex,cancer,kidney_injury,stroke,heart_failure,hepatic_failure,respiratory,brain_injury,Vital_Status)
write.csv(x,"Data.csv")


library(xgboost)  # the main algorithm
library(archdata) # for the sample dataset
library(caret)    # for the confusionmatrix() function (also needs e1071 package)
library(dplyr)  
library(adabag)
library(caret)
library(e1071)
library(ipred)
library(randomForest)

X=read.csv("Data.csv");head(X)




library(caTools)
#set.seed(0)
sp=sample.split(X,SplitRatio = 0.7)
tr=subset(X,sp==TRUE)
trx=tr[,-16]
try=tr[,16]
ts=subset(X,sp==FALSE);head(ts)
tsx=ts[,-16];head(tsx)
tsy=ts[,16];head(tsy)

#Logistic Regression
g1=glm(Vital_Status~.,data=tr,family=binomial)
summary(g1)
prob=predict(g1,tsx,type = "response")
prob=as.data.frame(prob);prob
prob=round(prob,2);prob
p=ifelse(prob>0.5,1,0)
t=table(tsy,p);t
(accuracy=sum(diag(t))/sum(t))
(recall=diag(t)/rowSums(t))
(precision=diag(t)/colSums(t))
(F=(2*precision*recall)/(precision+recall))




#RandomForest
rf=randomForest(Vital_Status~.,data=tr)
p_y=predict(rf,tsx);p_y
py=ifelse(p_y>0.5,1,0)
t1=table(tsy,py);t1
(accuracy=sum(diag(t1))/sum(t1))
(recall=diag(t1)/rowSums(t1))
(precision=diag(t1)/colSums(t1))
(F=(2*precision*recall)/(precision+recall))


#Linear Discriminant Analysis
library("MASS")
ld=lda(Vital_Status~.,data=tr)
summary(ld)
ldp=predict(ld,tsx);ldp
t2=table(tsy,ldp$class);t2
(accuracy=sum(diag(t2))/sum(t2))
(recall=diag(t2)/rowSums(t2))
(precision=diag(t2)/colSums(t2))
(F=(2*precision*recall)/(precision+recall))


#decision tree

library(rpart)
library(rpart.plot)
reg=rpart(formula = Vital_Status~.,data=tr,method="class")
rpart.plot(reg,extra=1,type=1)
pg=predict(reg,tsx,type='class');pg
t3=table(tsy,pg);t3
(accuracy=sum(diag(t3))/sum(t3))
(recall=diag(t3)/rowSums(t3))
(precision=diag(t3)/colSums(t3))
(F=(2*precision*recall)/(precision+recall))


library(class)
ky=knn(trx,tsx,try,k=3)
t4=table(tsy,ky);t4
(accuracy=sum(diag(t4))/sum(t4))
(recall=diag(t4)/rowSums(t4))
(precision=diag(t4)/colSums(t4))
(F=(2*precision*recall)/(precision+recall))



#SVM
library(e1071)
tr$Vital_Status= as.factor(tr$Vital_Status)
sv=svm(Vital_Status~.,data=tr)
summary(sv)
pq=predict(sv,tsx);pq
as.data.frame(pq)
t6=table(tsy,pq);t6
(accuracy=sum(diag(t6))/sum(t6))
(recall=diag(t6)/rowSums(t6))
(precision=diag(t6)/colSums(t6))
(F=(2*precision*recall)/(precision+recall))




