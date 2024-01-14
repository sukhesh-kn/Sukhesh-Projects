A2=read.csv("INFY.csv");A2
X2=A2$Close.Price;X2
n2=length(X2);n2
xt=ts(X2)
plot.ts(xt,main="Closing price of Infosys",xlab="Time points(in days)",ylab="Closing Price")

train=xt[249:2683]

##TREND
library(Kendall)
summary(MannKendall(train))
pvalue=2.22e-16
if(pvalue<0.05)
  cat("\n There is trend \n")else
    cat("\n There is no trend \n")
library(forecast)
auto.arima(xt,d=NA)
v=var(train);v
d1=diff(train)
v1=var(d1);v1
d2=diff(d1);d2
v2=var(d2);v2
acf(d2)
pacf(d2)
fit1=arima(train,order=c(13,1,2))
fi1=fitted(fit1)
b1=Box.test(fit1$residual,lag=2*sqrt(length(train)))
b1

res=fit1$residuals
par(mfrow=c(2,2))
res
acf(res,main="ACF of adjusted Close price")
acf(res^2,main="ACF of adjusted squared Close price")
pacf(res,main="PACF of adjusted Close price")
pacf(res^2,main="PACF of adjusted squared Close price")
fore=forecast(fit1,h=365)
plot(fore)
accuracy(fit1)






#
library(fGarch)
library(tseries)
gft1=garchFit(~garch(1,0),data=res,cond.dist=c("snorm"))
summary(gft1)
rs1=residuals(gft1, standardize=T)
l1=Box.test(rs1,lag=round(2*sqrt(length(train))),type="Ljung-Box");l1

pr=predict(gft1, n.ahead =365 , trace = TRUE, mse = c("cond","uncond"),plot=TRUE, nx=NULL, crit_val=NULL, conf=0.95);pr
pr
pr1=data.frame(pr)
attach(pr1)
PointFor1=pr1[,1]
print(pr1)
testpred=forecast(fit1,h=365);testpred
testpred1=data.frame(testpred)
attach(testpred1)
PointFor=testpred1[,1]
print(testpred1)
foreca=PointFor+PointFor1;foreca
for1=data.frame(foreca)
plot(foreca)
foreca=ts(foreca,frequency=365,start=c(2684,1))
head(foreca)

final=c(train,foreca)
length(final)
plot(final,type="l",xlim=c(0,2900))
lines(1:2435,train,col=1)
lines(2436:2800,foreca,col="red")
accuracy(fit1)

#Hybrid Model
library(forecastHybrid)
mod1=hybridModel(train,model="an")
fore=forecast(mod1,h=269)
plot(fore)
accuracy(fore)









A4=read.csv("HINDUNILVR.csv");A4
X4=A4$Close.Price;X4
xt=ts(X4,frequency=269,start=c(2012))
plot.ts(xt,main="Closing price of HINDUNILVR",xlab="Time points(in days)",ylab="Closing Price")

train=xt[249:2683]

##TREND
library(Kendall)
summary(MannKendall(train))
pvalue=0.00000000000000222
if(pvalue<0.05)
  cat("\n There is trend \n")else
    cat("\n There is no trend \n")
library(forecast)
auto.arima(xt,d=NA)
v=var(train);v
d1=diff(train)
v1=var(d1);v1
d2=diff(d1);d2
v2=var(d2);v2
acf(d2)
pacf(d2)
fit1=arima(train,order=c(13,1,1))
fi1=fitted(fit1)
b1=Box.test(fit1$residual,lag=2*sqrt(length(train)))
b1

res=fit1$residuals
par(mfrow=c(2,2))
res
acf(res,main="ACF of adjusted Close price")
acf(res^2,main="ACF of adjusted squared Close price")
pacf(res,main="PACF of adjusted Close price")
pacf(res^2,main="PACF of adjusted squared Close price")
library(fGarch)
library(tseries)
gft1=garchFit(~garch(1,2),data=res,cond.dist=c("snorm"))
summary(gft1)
rs1=residuals(gft1, standardize=T)
l1=Box.test(rs1,lag=round(2*sqrt(length(train))),type="Ljung-Box");l1
ms=c("1,1","1,2","1,3","1,4","2,1","2,4","3,1","4,1")
aic=c(8.223526,8.220453,8.215280,8.216585,8.225122,8.217406,8.226413,8.228007)
pvl=c(0.2502,0.253,0.2322,0.2326,0.2499,0.2326,0.25,0.2499)
B=data.frame("Model for HDFC"=c(1:8),"m,s"=ms,"AIC"=aic,"p value"=pvl);B

pr=predict(gft1, n.ahead =365 , trace = TRUE, mse = c("cond","uncond"),plot=TRUE, nx=NULL, crit_val=NULL, conf=0.95);pr
pr
pr1=data.frame(pr)
attach(pr1)
PointFor1=pr1[,1]
print(pr1)
testpred=forecast(fit1,h=365);testpred
testpred1=data.frame(testpred)
attach(testpred1)
PointFor=testpred1[,1]
print(testpred1)
foreca=PointFor+PointFor1;foreca
for1=data.frame(foreca)
plot(foreca)
foreca=ts(foreca,frequency=269,start=c(2684,1))
head(foreca)

final=c(train,foreca)
length(final)
plot(final,type="l",xlim=c(0,2900))
lines(1:2435,train,col=1)
lines(2436:2800,foreca,col="red")
accuracy(fit1)
t=c("Reliance","HDFC","Infosys","TCS","HINDUNILVR")
me=c(0.6391603,0.305811,-0.3541078,1.020076,0.7807012)
rmse=c(23.20465,17.99515,42.88895,52.6126,21.00564)
mae=c(15.23096,11.96699,19.58406,26.87696,12.66731)
mpe=c(0.03000693,0.02833256,-0.03754363,0.01995384,0.05957811)
mape=c(1.278448,1.025046,1.227878,1.157546,1.02444)
mase=c(0.9980176,1.005161,0.9961504,0.9961671,1.000977)
acfi=c(-0.001274759,-0.0005426328,-0.0002494552,-0.0003209411,-0.001474009)
AcTable=data.frame("Company Name"=t,"ME"=me,"RMSE"=rmse,"MAE"=mae,"MPE"=mpe,"MAPE"=mape,"MASE"=mase,"ACFI"=acfi);AcTable

#Hybrid Method

library(forecastHybrid)
train=ts(train,frequency=269,start=(2011))
mod1=hybridModel(train,model="an")
fore=forecast(mod1,h=269)
fore1=ts(fore,frequency=269,start=c(2021))
plot(fore,xlim=c(2011,2023),xlab="Time",ylab="Closing Price")
accuracy(fore)



A1=read.csv("Reliance.csv",header=TRUE)
B=ts(A1$Close.Price[249:2683],frequency=269,start=c(2012))
B1=ts(B[1077:1614],frequency=269,start=c(2016))
B2=ts(B[2152:2421],frequency=269,start=c(2020))
plot(B,xlim=c(2012,2023),ylab="Closing price")
lines(B1,col="red")
lines(B2,col="blue")


A2=read.csv("HDFC.csv",header=TRUE)
C=ts(A2$Close.Price[249:2683],frequency=269,start=c(2012))
C1=ts(C[1077:1614],frequency=269,start=c(2016))
C2=ts(C[2152:2421],frequency=269,start=c(2020))
plot(C,xlim=c(2012,2023),ylab="Closing price")
lines(C1,col="red")
lines(C2,col="blue")


A3=read.csv("INFY.csv",header=TRUE)
D=ts(A3$Close.Price[249:2683],frequency=269,start=c(2012))
D1=ts(D[1077:1614],frequency=269,start=c(2016))
D2=ts(D[2152:2421],frequency=269,start=c(2020))
plot(D,xlim=c(2012,2023),ylab="Closing price")
lines(D1,col="red")
lines(D2,col="blue")

A4=read.csv("TCS.csv",header=TRUE)
E=ts(A4$Close.Price[249:2683],frequency=269,start=c(2012))
E1=ts(E[1077:1614],frequency=269,start=c(2016))
E2=ts(E[2152:2421],frequency=269,start=c(2020))
plot(E,xlim=c(2012,2023),ylab="Closing price")
lines(E1,col="red")
lines(E2,col="blue")


A5=read.csv("HINDUNILVR.csv",header=TRUE)
F=ts(A5$Close.Price[249:2683],frequency=269,start=c(2012))
F1=ts(F[1077:1614],frequency=269,start=c(2016))
F2=ts(F[2152:2421],frequency=269,start=c(2020))
plot(F,xlim=c(2012,2023),ylab="Closing price")
lines(F1,col="red")
lines(F2,col="blue")





















#SVM for reliance
Rel=read.csv("Reliance.csv")
head(Rel)
daily_data=unclass(Rel)
head(daily_data)
days=249:2683
DF <- data.frame(days,daily_data$Close.Price[249:2683])
colnames(DF)<-c("x","y")
DF

library(e1071)
library(forecast)
library(nnfor)

svmodel <- svm(y ~ x,data=DF, type="eps-regression",kernel="radial",cost=10000, gamma=0.1)

nd <- 249:2683
length(nd)
#compute forecast for all the 156 months 
prognoza <- predict(svmodel, newdata=data.frame(x=nd))
accuracy(prognoza,DF$y)
fore=ts(prognoza,frequency=269,start=c(2012))
fore
length(fore)

nd1=2684:2952
prognoza1 <- predict(svmodel, newdata=data.frame(x=nd1))
fore1=ts(prognoza1,frequency = 269,start=c(2021.1))

tt=ts(daily_data$Close.Price,frequency=269,start=c(2012))
tt
plot.ts(tt)

y1=ts(DF$y,frequency=269,start=c(2012))

plot.ts(y1, type="l",col="black",ylim=c(400,3500), xlim=c(2011,2022),main="SVM-time series model along with the Original Values")
par(new=TRUE)
plot(fore, type="l",col="red", ylim=c(400,3500), xlim=c(2011,2022))
lines(fore1,type="l",col="blue")

plot(fore1, type="l",col="red", ylim=c(400,3500), xlim=c(2011,2022),ylab="Closing price")
plot.ts(y1, type="l",col="black",ylim=c(400,3500), xlim=c(2011,2022),ylab="Closing price",main="SVM-time series model")
par(new=TRUE)
plot(fore1, type="l",col="red", ylim=c(400,3500), xlim=c(2011,2022),ylab="Closing price")


library(tkrplot)
library(sm)
library(rpanel)
r=read.csv("Reliance.csv")
sm.density(r,model="Normal",panel=TRUE, xlim=c(2012,2020),ylim=c(1500,2000))
