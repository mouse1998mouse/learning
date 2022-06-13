# session

## 分布式下session的存储机制

服务端：
- 将session通过序列化跟反序列化机制，持久化到硬盘中
- 将session放在cookie中
- ==使用redis来进行存储==，每次登录使用Redis在存储会话信息。并且设置过期时间
- session复制。将这台机上上的session同步到其他服务器。
- session绑定，例如使用nginx进行绑定，这样客户端就会访问同一个服务器进行处理

