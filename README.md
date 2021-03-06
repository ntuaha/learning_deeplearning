# learning_deeplearning

## 目的
一步一步按照書上的說法，從 0 開始建立相關細節，並從中學習許多實作上可以改進的與發想的地方．全書主要一共有 439 pages．每次閱讀約 80 頁，應該六次就能完成全部內容．

## 參考用書
[Deep Learning 3：用Python進行深度學習框架的開發實作](https://www.books.com.tw/products/0010887759?gclid=Cj0KCQjw1a6EBhC0ARIsAOiTkrHlvtopTx7NQStp6X7vkPwGtPdXWNm_gQ-MT3ayQNHX-ePDo_RO0vcaAlaZEALw_wcB)


## 進度

### 20210508 - P101

1. 新增Config，好方便快速建立有沒有反向傳播的資訊
2. 透過 with 的語法，好讓 Config 可以切換更快速
3. add Name attr in a Variable class
4. 讓 Variable 有 shape attr
5. 逐步調整 Variable 看起來像似 numpy P113 開始
6. 讓所有輸入參數都要轉換成 Variable

### 20210507 - P101
參考工具 https://pypi.org/project/memory-profiler/
調整 python kernel for jupyter
```shell
ipython kernel install --name "local-venv" --user
# for pythoh 3.9
# on correct environment
ipython kernel install --name 'python3.9' --user
```
確認相關記憶體使用狀況，參考[教學文件](https://coderzcolumn.com/tutorials/python/how-to-profile-memory-usage-in-python-using-memory-profiler)
```shell
 mprof run tests/memory_usage.py   
 mprof plot [filename]
```
1. 記憶體節省使用弱參照 P97
2. 記憶體節省降低中間計算的 grad P99

### 20210506 - P58 - P96
1. 調整 Variable 
    a. 修改變數反向傳播 P67
    b. 變數再接收回傳 grad 的時候，必須接收所有當次回送回來的 grad P71
    c. 需要增加清除 grad function 在每次反向傳播之前 P73
    d. 增加 generation 在變數初始化，並在指定creator的時候，設定為該creator.generation + 1 P83
    e. 變數進行 backward 計算的時候，應該要按照generation 從大到小計算，並且要踢出重複的 P87
2. 調整 Function
    a. Function __call__ 吃 input array, 輸出時多個 array, 單個就單元素
    b. 設定函數 generation 為所有輸入參數的最大值

### 20210501 - P58
### 20210430 - P1 - P57

