# 防止sql注入
### 使用占位符，采用预编译语句
### 进行字符串过滤 
如：
```
String inj_str = "'|and|exec|insert|select|delete|update|

count|*|%|chr|mid|master|truncate|char|declare|;|or|-|+|,";
```