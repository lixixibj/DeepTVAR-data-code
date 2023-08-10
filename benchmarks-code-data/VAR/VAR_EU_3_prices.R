
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
#accuracy mse
mse_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  mse<-(outsample-forecasts)*(outsample-forecasts)
  return(mse)
}

#mape
mape_cal <- function(outsample, forecasts) {
  #Used to estimate sMAPE
  outsample <- as.numeric(outsample) ; forecasts<-as.numeric(forecasts)
  mape <- (abs(outsample-forecasts)*100)/(abs(outsample))
  return(mape)
}
#MSIS
#calculate msis of every time t
msis_cal<-function(insample, outsample,forecasts.lower,forecasts.upper,a=0.05){
  #
  frq <- stats::frequency(insample)
  m<-c()
  for (j in (frq+1):length(insample)){
    m <- c(m, abs(insample[j]-insample[j-frq]))
  }
  masep<-mean(m)
  #
  b<-c()
  for (i in 1:length(outsample)){
    U.subtract.L<-forecasts.upper[i]-forecasts.lower[i]
    if(outsample[i]<forecasts.lower[i]){
      r<-(2/a)*(forecasts.lower[i]-outsample[i])
    }else{
      r<-0
    }
    if(outsample[i]>forecasts.upper[i]){
      q<-(2/a)*(outsample[i]-forecasts.upper[i])
    }else{
      q<-0
    }
    b<-c(b, U.subtract.L+r+q)
  }
  return(b/masep)
}

mis_cal<-function(outsample,forecasts.lower,forecasts.upper,a=0.05){
  #
  #frq <- stats::frequency(insample)
  # m<-c()
  # for (j in (frq+1):length(insample)){
  #   m <- c(m, abs(insample[j]-insample[j-frq]))
  # }
  # masep<-mean(m)
  #
  b<-c()
  for (i in 1:length(outsample)){
    U.subtract.L<-forecasts.upper[i]-forecasts.lower[i]
    if(outsample[i]<forecasts.lower[i]){
      r<-(2/a)*(forecasts.lower[i]-outsample[i])
    }else{
      r<-0
    }
    if(outsample[i]>forecasts.upper[i]){
      q<-(2/a)*(outsample[i]-forecasts.upper[i])
    }else{
      q<-0
    }
    b<-c(b, U.subtract.L+r+q)
  }
  return(b)
}
#VAR model
#var model
var.prediction<- function(diff.data,lag,num.of.ts) {
  var.m <- vars::VAR(diff.data, p=lag)
  re=summary(var.m)
  library(vars)
  library(matrixcalc)
  coeffs=Acoef(var.m)
  forecasts <- predict(var.m,n.ahead =horizons,ci=0.95)
  plot(forecasts)
  
  #back-transformation of differenced series
  #interval predition 
  
  #put all A1,...,Ap together
  A=c()
  for (i in 1:lag) {
    A=cbind(A,coeffs[[i]])
  }
  mp=num.of.ts*lag
  #identy matrix
  identy.m=diag(mp-num.of.ts)
  zero.m=matrix(0, (mp-num.of.ts), num.of.ts)
  added_m=cbind(identy.m,zero.m)
  big.A.m=unname(rbind(A,added_m))
  
  #put sigma and zero together
  zero.m1=matrix(0, num.of.ts, (mp-num.of.ts))
  zero.m2=matrix(0, (mp-num.of.ts),mp)
  big.sigma=unname(rbind(cbind(re$covres,zero.m1),zero.m2))
  J=cbind(diag(num.of.ts),matrix(0, num.of.ts,(mp-num.of.ts)))
  sum.of.A.matrix<-function(big.A.m,k,h,mp){
    A.sum=matrix(0, mp,mp)
    for (l in k:h) {
      A.sum=A.sum+matrix.power(big.A.m, (l-k))
    }
    return(A.sum)
  }
  sigma.m=matrix(0,horizons,num.of.ts)
  
  for (h in 1:horizons) {
    #cal 
    sum.m=matrix(0, mp,mp)
    for (k in 1:h) {
      #multicative
      sum.m=sum.m+sum.of.A.matrix(big.A.m,k,h,mp)%*%big.sigma%*%t(sum.of.A.matrix(big.A.m,k,h,mp))
    }
    var_cov=J%*%sum.m%*%t(J)
    #
    for (ts.index in 1:num.of.ts) {
      sigma.m[h,ts.index]=sqrt(var_cov[ts.index,ts.index])
    }
  }
  return(list(ff=forecasts,sigma=sigma.m))
}

diff.dataframe<-function(original.data){
  r=dim(original.data)[1]
  c=dim(original.data)[2]
  diff.data=original.data[1:(r-1),]
  for (n in 1:c) {
    diff.data[,n]=diff(original.data[,n])
  }
  
  return(diff.data)
  #return(list(ff=diff.data,dd=1))
}

num.of.ts=3
horizons=12
num.of.forecast=20
freq=12
level.value=95
all_data=read.csv('/Users/xixili/Dropbox/DeepTVAR-code/benchmarks-all/eu-prices.csv')
all_data=all_data[,2:4]

#log processing
m=dim(all_data)[2]
T=dim(all_data)[1]-1
for (c in 1:m) {
  all_data[,c]=log(all_data[,c])
}

par(mfrow=c(3,3))
ts.plot(all_data$electricity)
ts.plot(diff(all_data$electricity))
acf(all_data$electricity)
ts.plot(all_data$natural.gas)
ts.plot(diff(all_data$natural.gas))
acf(diff(all_data$natural.gas))
ts.plot(all_data$petrol)
ts.plot(diff(all_data$petrol))
acf(diff(all_data$petrol))

# Set random seed
set.seed(100)
len=dim(all_data)[1]
test_len=horizons+num.of.forecast-1
train_len=len-test_len
lag_order=2

mse.accuracy.m.ts1=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mse.accuracy.m.ts2=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mse.accuracy.m.ts3=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mape.accuracy.m.ts1=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mape.accuracy.m.ts2=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mape.accuracy.m.ts3=matrix(NA,nrow = num.of.forecast,ncol = horizons)

mis.accuracy.m.ts1=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mis.accuracy.m.ts2=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mis.accuracy.m.ts3=matrix(NA,nrow = num.of.forecast,ncol = horizons)

msis.accuracy.m.ts1=matrix(NA,nrow = num.of.forecast,ncol = horizons)
msis.accuracy.m.ts2=matrix(NA,nrow = num.of.forecast,ncol = horizons)
msis.accuracy.m.ts3=matrix(NA,nrow = num.of.forecast,ncol = horizons)
mainDir='/Users/xixili/Dropbox/DeepTVAR-code/benchmarks-all/eu-3-prices-var-model-p3'
for (f in 1:num.of.forecast) {
  print('num')
  print(f)
  point.forecasts.m=matrix(data=0,nrow = 3,ncol = horizons)
  lower.forecasts.m=matrix(data=0,nrow = 3,ncol = horizons)
  upper.forecasts.m=matrix(data=0,nrow = 3,ncol = horizons)
  #var model
  b=f
  e=f+train_len-1
  #fit tv-var model
  #bv <- bvar.sv.tvp(as.matrix(all_data[b:e,]))
  original.data=all_data[b:e,]
  diff.data=diff.dataframe(original.data)
  re=var.prediction(diff.data,lag_order,num.of.ts)
  forecasts=re$ff
  sigma.m=re$sigma
  for (ts in 1:num.of.ts) {
    if (ts==1){
      x=ts(all_data[b:e,1],frequency = freq)
      xx=ts(all_data[(1+e):(e+horizons),1],frequency = freq)
      point.forecasts.diff=forecasts$fcst$electricity[,1]
      point.forecasts=cumsum(point.forecasts.diff)+all_data$electricity[e]
      mse<-mse_cal(xx,point.forecasts)
      mape<-mape_cal(xx,point.forecasts)
      #interval prediction
      lower.prediction=point.forecasts-1.96*sigma.m[,ts]
      upper.prediction=point.forecasts+1.96*sigma.m[,ts]
      msis=msis_cal(x, xx,lower.prediction,upper.prediction,a=0.05)
      mis=mis_cal(xx,lower.prediction,upper.prediction,a=0.05)
      mis.accuracy.m.ts1[f,]=mis
      
      mse.accuracy.m.ts1[f,]=mse
      mape.accuracy.m.ts1[f,]=mape
      msis.accuracy.m.ts1[f,]=msis
      
    }else if (ts==2) {
      x=ts(all_data[b:e,2],frequency = freq)
      xx=ts(all_data[(1+e):(e+horizons),2],frequency = freq)
      point.forecasts.diff=forecasts$fcst$natural.gas[,1]
      point.forecasts=cumsum(point.forecasts.diff)+all_data$natural.gas[e]
      mse<-mse_cal(xx,point.forecasts)
      mape<-mape_cal(xx,point.forecasts)
      #interval prediction
      lower.prediction=point.forecasts-1.96*sigma.m[,ts]
      upper.prediction=point.forecasts+1.96*sigma.m[,ts]
      msis=msis_cal(x, xx,lower.prediction,upper.prediction,a=0.05)
      mis=mis_cal(xx,lower.prediction,upper.prediction,a=0.05)
      mis.accuracy.m.ts2[f,]=mis
      
      mse.accuracy.m.ts2[f,]=mse
      mape.accuracy.m.ts2[f,]=mape
      msis.accuracy.m.ts2[f,]=msis
    } else if (ts==3) {
      x=ts(all_data[b:e,3],frequency = freq)
      xx=ts(all_data[(1+e):(e+horizons),3],frequency = freq)
      point.forecasts.diff=forecasts$fcst$petrol[,1]
      point.forecasts=cumsum(point.forecasts.diff)+all_data$petrol[e]
      mse<-mse_cal(xx,point.forecasts)
      mape<-mape_cal(xx,point.forecasts)
      #interval prediction
      lower.prediction=point.forecasts-1.96*sigma.m[,ts]
      upper.prediction=point.forecasts+1.96*sigma.m[,ts]
      msis=msis_cal(x, xx,lower.prediction,upper.prediction,a=0.05)
      mis=mis_cal(xx,lower.prediction,upper.prediction,a=0.05)
      mis.accuracy.m.ts3[f,]=mis
      
      mse.accuracy.m.ts3[f,]=mse
      mape.accuracy.m.ts3[f,]=mape
      msis.accuracy.m.ts3[f,]=msis
    }
  }
  #create dir
  dir.create(file.path(mainDir, as.character(f)),showWarnings = FALSE)
  #setwd(file.path(mainDir, as.character(f)))
  path= paste(mainDir,as.character(f),sep = '/')
  write.csv(point.forecasts.m,file =paste(path,'VAR_point_forecasts.csv',sep = '/'))
  write.csv(lower.forecasts.m,file =paste(path,'VAR_lower_forecasts.csv',sep = '/'))
  write.csv(upper.forecasts.m,file =paste(path,'VAR_upper_forecasts.csv',sep = '/'))
}


print('ts')
print(1)
print('mse')
mse1=colMeans(mse.accuracy.m.ts1)
print(colMeans(mse.accuracy.m.ts1))
print('h1-6')
print(mean(mse1[1:6]))
print('h1-12')
print(mean(mse1[1:12]))

print('mape')
mape=colMeans(mape.accuracy.m.ts1)
print('h1-6')
print(mean(mape[1:6]))
print('h1-12')
print(mean(mape[1:12]))
print(colMeans(mape.accuracy.m.ts1))

print('msis')
print(colMeans(msis.accuracy.m.ts1))
msis=colMeans(msis.accuracy.m.ts1)
print('h1-6')
print(mean(msis[1:6]))
print('h1-12')
print(mean(msis[1:12]))

print('mis')
print(colMeans(mis.accuracy.m.ts1))
mis1=colMeans(mis.accuracy.m.ts1)
print('h1-6')
print(mean(mis1[1:6]))
print('h1-12')
print(mean(mis1[1:12]))



print('ts')
print(2)
print('mse')
mse2=colMeans(mse.accuracy.m.ts2)
print(colMeans(mse.accuracy.m.ts2))
print('h1-6')
print(mean(mse2[1:6]))
print('h1-12')
print(mean(mse2[1:12]))

print('mape')
mape2=colMeans(mape.accuracy.m.ts2)
print('h1-6')
print(mean(mape2[1:6]))
print('h1-12')
print(mean(mape2[1:12]))
print(colMeans(mape.accuracy.m.ts2))

print('msis')
print(colMeans(msis.accuracy.m.ts2))
msis2=colMeans(msis.accuracy.m.ts2)
print('h1-6')
print(mean(msis2[1:6]))
print('h1-12')
print(mean(msis2[1:12]))


print('mis')
print(colMeans(mis.accuracy.m.ts2))
mis2=colMeans(mis.accuracy.m.ts2)
print('h1-6')
print(mean(mis2[1:6]))
print('h1-12')
print(mean(mis2[1:12]))

print('ts')
print(3)
print('mse')
mse3=colMeans(mse.accuracy.m.ts3)
print(colMeans(mse.accuracy.m.ts3))
print('h1-6')
print(mean(mse3[1:6]))
print('h1-12')
print(mean(mse3[1:12]))

print('mape')
mape3=colMeans(mape.accuracy.m.ts3)
print('h1-6')
print(mean(mape3[1:6]))
print('h1-12')
print(mean(mape3[1:12]))
print(colMeans(mape.accuracy.m.ts3))

print('msis')
print(colMeans(msis.accuracy.m.ts3))
msis3=colMeans(msis.accuracy.m.ts3)
print('h1-6')
print(mean(msis3[1:6]))
print('h1-12')
print(mean(msis3[1:12]))

print('mis')
print(colMeans(mis.accuracy.m.ts3))
mis3=colMeans(mis.accuracy.m.ts3)
print('h1-6')
print(mean(mis3[1:6]))
print('h1-12')
print(mean(mis3[1:12]))


