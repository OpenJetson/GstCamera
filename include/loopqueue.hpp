/***************************************
@Copyright(C) All rights reserved.
@Author CHENG Liang
@Date   2020.08
@File   loopqueue.hpp
***************************************/

#ifndef LOOPQUEUE_H
#define LOOPQUEUE_H

#include <iostream>
#include <string>

#include <thread>         // std::thread
#include <mutex>          // std::mutex

using namespace std;

template <typename T>
class LoopQueue{
private:
  T *queue;//存储用的数组
  int capacity;//存放个数
  int head;//指向队首
  int tail;//指向队尾

  std::mutex loopQueue_MT;

public:
  LoopQueue(int a);//无参构造
  LoopQueue();//有参构造
  ~LoopQueue();//析构
  bool isEmpty();//判断空
  int getSize();//返回个数
  bool push(T a);//入队
  bool pop();//出队
  bool pop2(T& t);
  T top();//显示队首
};

template<typename T>
LoopQueue<T>::LoopQueue(int a) :head(0), tail(0), capacity(a), queue(nullptr){
  queue = new T[capacity];
}

template<typename T>
LoopQueue<T>::LoopQueue() : LoopQueue(10){};

template<typename T>
LoopQueue<T>::~LoopQueue(){
  delete[] queue;
}

template<typename T>
bool LoopQueue<T>::isEmpty(){
  if (head == tail)
  return true;
  else
  return false;
}

template<typename T>
  int LoopQueue<T>::getSize(){
  return (tail - head + capacity) % capacity;
}

template<typename T>
bool LoopQueue<T>::push(T a){
  if ((tail - head + capacity) % capacity == (capacity-1))
  return false;

  while(1){
    if(loopQueue_MT.try_lock()){
          queue[tail] = a;
          tail = (tail + 1) % capacity;
          loopQueue_MT.unlock();
          break;
     }
    else{
        usleep(1);
    }
  }
  return true;
}

template<typename T>
bool LoopQueue<T>::pop(){
  if ((tail - head + capacity) % capacity == 0)
  return false;
  head = (head + 1) % capacity;
  return true;
}

template<typename T>
T LoopQueue<T>::top(){
   return queue[head%capacity];
}

template<typename T>
bool LoopQueue<T>::pop2(T& t){
   // T t = queue[head%capacity];
    if ((tail - head + capacity) % capacity == 0){
        return false;
    }
    while(1){
        if(loopQueue_MT.try_lock()){
            t =queue[head%capacity];
            head = (head + 1) % capacity;
            loopQueue_MT.unlock();
            break;
        }
        else
        {
            usleep(1);
        }
    }
    return true;
}

#endif // LOOPQUEUE_H