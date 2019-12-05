#ifndef _LOGWEIGHTPROB_HPP_
#define _LOGWEIGHTPROB_HPP_


#include <cmath>
#include <stdlib.h>

template <class FT>
class LogWeightProb{

  ///weight or prob in log representation
  FT log_wp;
  bool zero;

  //be careful, must be log2 to be compatible with getLog2
  inline FT my_log(FT x) const{
    return log2(x);
  }


  inline FT my_exp(FT x) const{
    return exp2(x);
  }

public:

  typedef FT FT_Type;
  
  //just put some arbitrary value log_wp to avoid unitialized value problem
  LogWeightProb():log_wp(0),zero(true){}


  LogWeightProb(FT wp){
    if(wp<0){
      std::cerr<<"Wrong weight/prob. Should be >=0"<<std::endl;
      exit(1);
    }
    zero=(wp==0);
    if(!zero)
      log_wp=my_log(wp);
    else
      log_wp=0;//just put some arbitrary value log_wp to avoid unitialized value problem
  }

  inline FT getWeightProb() const{
    if(zero)
      return 0;
    else
      return my_exp(log_wp);
  }

  inline FT getLog2() const{
    return log_wp;
  }

  inline FT setLog2(FT _log_wp) const {
	  log_wp=_log_wp;
  }


  inline void setLog2(FT l2){
    log_wp=l2;
    if(l2==l2)//l2 is not nan
      zero=false;
    else{
      std::cerr<<"Setting nan: something is wrong"<<std::endl;
      zero=true;
    }
  }

  inline bool isZero() const{
    return zero;
  }

  
  //This code is a modification of common.hpp in http://jveness.info/software/cts-v1.zip
  //given log(x) and log(y), compute log(x+y). uses the following identity:
  //log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x)))
  inline LogWeightProb<FT> operator+(LogWeightProb<FT> y) {
    LogWeightProb<FT> ret;//initialized 0
    if(!(zero&&y.zero)){
      ret.zero=false;
      if(zero)
	ret.log_wp=y.log_wp;
      else if(y.zero)
	ret.log_wp=this->log_wp;
      else{
        FT log_x = this->log_wp;
        FT log_y = y.log_wp;
        // ensure log_y >= log_x, can save some expensive log/exp calls
        if (log_x > log_y) {
          FT t = log_x; log_x = log_y; log_y = t;
        }

        ret.log_wp=log_y - log_x;
        // only replace log(1+exp(log(y)-log(x))) with log(y)-log(x)    
        // if the the difference is small enough to be meaningful
        if(ret.log_wp < 100)
          ret.log_wp= my_log(1.0 + my_exp(ret.log_wp));
        ret.log_wp+=log_x;
      }
    }
    return ret;
  }

  //given log(x) and log(y), compute log(x-y). uses the following identity:
  //log(x + y) = log(x) + log(1-exp(log(y)-log(x)))
  //simple version
  inline LogWeightProb<FT> operator-(LogWeightProb<FT> y) {
    LogWeightProb<FT> ret;//initialized 0

    if(!(y.zero&&zero)){
      ret.zero=false;
      
      if(y.zero)
	ret.log_wp=log_wp;
          
      else if(zero || (y.log_wp>log_wp) ){ 
	std::cerr<<"operator-: Can't store negative numbers in log representation"<<std::endl;
	exit(1);
      }

      else if (y.log_wp==log_wp)
	ret.zero=true;
      
      else
	ret.log_wp=this->log_wp + my_log(1.0 - my_exp(y.log_wp - this->log_wp));
    }
    return ret;
  }


  inline LogWeightProb<FT> operator*(LogWeightProb y) {
    LogWeightProb<FT> ret;//initialized 0
    if(!(y.zero||zero)){
      ret.zero=false;
      ret.log_wp=this->log_wp + y.log_wp;
    }
    return ret;
  }

  inline LogWeightProb<FT> operator/(LogWeightProb y) {
    if(y.zero) { 
      std::cerr<<"operator/: division by 0"<<std::endl;
      exit(1);
    }
   
    LogWeightProb<FT> ret;
    if(!zero){
      ret.zero=false;
      ret.log_wp=this->log_wp - y.log_wp;
    }
    return ret;
  }

  inline LogWeightProb<FT> operator/(double y) {
    if(y==0.0) { 
      std::cerr<<"operator/: division by 0"<<std::endl;
      exit(1);
    }
    LogWeightProb<FT> ret;
    if(!zero){
      ret.zero=false;
      ret.log_wp=this->log_wp - my_log(y);
    }
    return ret;
  }

  inline bool operator>=(const LogWeightProb & y) const {
    if(zero&&y.zero)
      return true;
    else if(zero)
      return false;
    else if(y.zero)
      return true;
    else
      return this->log_wp >= y.log_wp;
  }

  inline bool operator<(const LogWeightProb & y) const {
    return !(*this>=y);
  }

  inline bool operator>(const LogWeightProb & y) const {
    if(zero&&y.zero)
      return false;
    else if(zero)
      return false;
    else if(y.zero)
      return true;
    else
      return this->log_wp > y.log_wp;
  }

  inline LogWeightProb<FT> & operator+=(const LogWeightProb<FT> &rhs) {
    *this=*this+rhs;
    return *this;
  }
  
  inline bool operator<=(const LogWeightProb & y) const {
    return !(*this>y);
  }

  inline LogWeightProb<FT> & operator*=(const LogWeightProb<FT> &rhs) {
    *this=*this * rhs;
    return *this;
  }

  inline LogWeightProb<FT> & operator/=(const LogWeightProb<FT> &rhs) {
    *this=*this / rhs;
    return *this;
  }
 
  
};

#include <iostream>

  template <class FT>
  std::ostream& operator<<(std::ostream& os, const LogWeightProb<FT> & lwp)
  {
    os << lwp.getWeightProb() ;
    return os;
  }
 
template <class FT>
FT log2(LogWeightProb<FT> lwp)
{
  return lwp.getLog2();
}

template <class FT>
LogWeightProb<FT> lwp_exp2(FT l2)
{
  LogWeightProb<FT> ret;
  ret.setLog2(l2);
  return ret;
}
 
template <class FT>
LogWeightProb<FT> pow(LogWeightProb<FT> lwp,FT exponent)
{
  LogWeightProb<FT> ret;
  ret.setLog2(lwp.getLog2()*exponent);
  return ret;
}





#endif
